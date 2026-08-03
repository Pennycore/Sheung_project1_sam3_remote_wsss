from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile

from sam3_remote_wsss.cam.fusion import (
    fuse_cam_and_sam_with_stats,
    normalize_cams,
)
from sam3_remote_wsss.cam.dataset import PotsdamImageLevelDataset
from sam3_remote_wsss.config import ClassSpec, parse_config
from sam3_remote_wsss.evaluate_pseudo_labels import (
    compute_evaluation_metrics,
    main as evaluate_main,
)
from sam3_remote_wsss.evaluate_student import PatchRecord, stitch_parent
from sam3_remote_wsss.fusion import FusionCanvas
from sam3_remote_wsss.fuse_cam_sam import (
    _prepare_fusion_output,
    _require_complete_inputs,
)
from sam3_remote_wsss.generate_pseudo_labels import _merge_summaries
from sam3_remote_wsss.potsdam import (
    discover_potsdam_items,
    image_level_from_label,
    label_rgb_to_ids,
    read_label_rgb,
    read_rgbir_as_rgb,
)
from sam3_remote_wsss.prepare_potsdam_patches import (
    load_parent_split,
    prepare_patches,
)
from sam3_remote_wsss.repair_palette_label import repair_palette_label
from sam3_remote_wsss.train_cam import (
    _ensure_parent_disjoint,
    _f1_metrics,
    _parent_image_id,
    _prepare_training_output,
)
from sam3_remote_wsss.student.dataset import PotsdamGroundTruthSegDataset
from sam3_remote_wsss.train_student import (
    _ensure_parent_disjoint as ensure_student_parent_disjoint,
    segmentation_metrics,
)


CLASSES = (
    ClassSpec(1, "impervious_surface", (255, 255, 255), ("impervious surface",)),
    ClassSpec(2, "building", (0, 0, 255), ("building",)),
    ClassSpec(3, "low_vegetation", (0, 255, 255), ("low vegetation",)),
    ClassSpec(4, "tree", (0, 255, 0), ("tree",)),
    ClassSpec(5, "car", (255, 255, 0), ("car",)),
)


class PotsdamMappingTests(unittest.TestCase):
    def test_noisy_palette_label_is_repaired_with_distance_guard(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "noisy.tif"
            output_path = root / "repaired.tif"
            noisy = np.asarray(
                [
                    [[250, 4, 3], [4, 5, 245]],
                    [[5, 246, 247], [247, 248, 5]],
                ],
                dtype=np.uint8,
            )
            colors = np.asarray(
                [
                    [255, 0, 0],
                    [0, 0, 255],
                    [0, 255, 255],
                    [255, 255, 0],
                ],
                dtype=np.uint8,
            )
            names = ["background", "building", "low_vegetation", "car"]
            tifffile.imwrite(input_path, noisy, photometric="rgb")

            report = repair_palette_label(
                input_path,
                output_path,
                names,
                colors,
                max_distance=20.0,
                chunk_rows=1,
            )

            np.testing.assert_array_equal(
                tifffile.imread(output_path),
                colors[np.asarray([[0, 1], [2, 3]])],
            )
            self.assertEqual(report["pixels_over_threshold"], 0)
            self.assertEqual(
                report["pixel_counts"],
                {name: 1 for name in names},
            )

            guarded_output = root / "guarded.tif"
            with self.assertRaisesRegex(ValueError, "Refusing palette repair"):
                repair_palette_label(
                    input_path,
                    guarded_output,
                    names,
                    colors,
                    max_distance=1.0,
                )
            self.assertFalse(guarded_output.exists())

    def test_background_maps_to_zero_and_unknown_stays_ignore(self) -> None:
        label = np.array(
            [[[255, 0, 0], [0, 0, 255], [12, 34, 56]]],
            dtype=np.uint8,
        )

        ids = label_rgb_to_ids(
            label,
            CLASSES,
            ignore_index=255,
            background_colors=((255, 0, 0),),
        )

        np.testing.assert_array_equal(ids, np.array([[0, 2, 255]], dtype=np.uint8))

    def test_image_level_thresholds(self) -> None:
        label = np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8)
        label[0, :2] = (255, 255, 0)

        self.assertEqual(image_level_from_label(label, CLASSES)["car"], 1)
        self.assertEqual(
            image_level_from_label(label, CLASSES, min_class_pixels=3)["car"],
            0,
        )
        self.assertEqual(
            image_level_from_label(
                label,
                CLASSES,
                min_class_pixels=3,
                class_min_pixels={"car": 2},
            )["car"],
            1,
        )


class PseudoLabelPolicyTests(unittest.TestCase):
    def test_shard_summaries_merge_without_losing_existing_items(self) -> None:
        merged = _merge_summaries(
            [
                {"image_id": "patch_b", "kept_masks": 1},
                {"image_id": "patch_a", "kept_masks": 2},
            ],
            [
                {"image_id": "patch_b", "kept_masks": 3},
                {"image_id": "patch_c", "kept_masks": 4},
            ],
        )

        self.assertEqual(
            merged,
            [
                {"image_id": "patch_a", "kept_masks": 2},
                {"image_id": "patch_b", "kept_masks": 3},
                {"image_id": "patch_c", "kept_masks": 4},
            ],
        )

    def test_prompted_background_config_is_parsed(self) -> None:
        raw = _config(Path("/tmp/potsdam"))
        raw["background_prompting"] = {
            "enabled": True,
            "prompts": ["clutter in aerial imagery"],
            "score_threshold": 0.6,
            "min_mask_area": 16,
            "max_mask_area_ratio": 0.5,
            "conflict_margin": 0.04,
        }

        config = parse_config(raw)

        self.assertTrue(config.background_prompting.enabled)
        self.assertEqual(
            config.background_prompting.prompts,
            ("clutter in aerial imagery",),
        )
        self.assertEqual(config.background_prompting.score_threshold, 0.6)
        self.assertEqual(config.background_prompting.conflict_margin, 0.04)

    def test_uncovered_pixels_stay_ignored(self) -> None:
        canvas = FusionCanvas(height=2, width=3, ignore_index=255, uncovered_label=255)
        canvas.add_mask(
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
            class_id=2,
            score=0.8,
            x0=0,
            y0=0,
        )

        np.testing.assert_array_equal(
            canvas.result(),
            np.array([[2, 255, 255], [255, 2, 255]], dtype=np.uint8),
        )

    def test_prompted_background_fills_uncovered_and_ignores_conflicts(self) -> None:
        canvas = FusionCanvas(height=1, width=4, ignore_index=255, uncovered_label=255)
        canvas.add_mask(
            np.array([[1, 1]], dtype=np.uint8),
            class_id=2,
            score=0.8,
            x0=0,
            y0=0,
        )
        canvas.add_background_mask(
            np.array([[1, 0, 1, 1]], dtype=np.uint8),
            score=0.7,
            x0=0,
            y0=0,
        )
        canvas.add_background_mask(
            np.array([[0, 1, 0, 0]], dtype=np.uint8),
            score=0.78,
            x0=0,
            y0=0,
        )

        np.testing.assert_array_equal(
            canvas.result(background_conflict_margin=0.03),
            np.array([[2, 255, 0, 0]], dtype=np.uint8),
        )

    def test_strict_and_labeled_metrics_report_coverage(self) -> None:
        confusion = np.array([[1, 0], [0, 1]], dtype=np.int64)
        gt_count = np.array([1, 3], dtype=np.int64)
        labeled_gt_count = np.array([1, 1], dtype=np.int64)

        class Config:
            classes = (ClassSpec(1, "foreground", (255, 255, 255), ("foreground",)),)

        metrics = compute_evaluation_metrics(
            confusion,
            gt_count,
            labeled_gt_count,
            Config(),
        )

        self.assertEqual(metrics["class_iou"]["foreground"], 1 / 3)
        self.assertEqual(metrics["labeled_class_iou"]["foreground"], 1.0)
        self.assertEqual(metrics["labeled_coverage"], 0.5)
        self.assertEqual(metrics["per_class_labeled_coverage"]["foreground"], 1 / 3)


class CAMFusionTests(unittest.TestCase):
    def test_fusion_output_rejects_changed_or_untracked_inputs(self) -> None:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "fusion"
            provenance = {"version": 1, "sam": "first"}

            _prepare_fusion_output(output_dir, provenance)
            _prepare_fusion_output(output_dir, provenance)
            with self.assertRaisesRegex(ValueError, "inputs or settings changed"):
                _prepare_fusion_output(
                    output_dir,
                    {"version": 1, "sam": "second"},
                )

            legacy_dir = Path(temporary) / "legacy"
            (legacy_dir / "pseudo_labels").mkdir(parents=True)
            (legacy_dir / "pseudo_labels" / "patch.png").write_bytes(b"png")
            with self.assertRaisesRegex(FileExistsError, "without provenance"):
                _prepare_fusion_output(legacy_dir, provenance)

    def test_complete_fusion_inputs_are_required(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sam_dir = root / "sam"
            cam_dir = root / "cam"
            sam_dir.mkdir()
            cam_dir.mkdir()
            (sam_dir / "patch_a.png").write_bytes(b"png")
            (cam_dir / "patch_b.npz").write_bytes(b"npz")

            with self.assertRaisesRegex(
                FileNotFoundError,
                "missing SAM3=1.*missing CAM=1",
            ):
                _require_complete_inputs(
                    {"patch_a", "patch_b"},
                    sam_dir,
                    cam_dir,
                )

            (cam_dir / "patch_a.npz").write_bytes(b"npz")
            (sam_dir / "patch_b.png").write_bytes(b"png")
            _require_complete_inputs(
                {"patch_a", "patch_b"},
                sam_dir,
                cam_dir,
            )

    def test_cam_normalization_is_per_class(self) -> None:
        cams = np.array(
            [
                [[0.0, 2.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        normalized = normalize_cams(cams)

        np.testing.assert_allclose(
            normalized[0],
            np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(normalized[1], np.zeros((2, 2), dtype=np.float32))

    def test_cam_sam_fusion_excludes_background_and_keeps_supported_sam(self) -> None:
        sam = np.array([[2, 255, 255, 3]], dtype=np.uint8)
        cams = np.array(
            [
                [[0.1, 0.8, 0.1, 0.9]],
                [[0.9, 0.1, 0.1, 0.4]],
            ],
            dtype=np.float32,
        )

        fused, stats = fuse_cam_and_sam_with_stats(
            sam_label=sam,
            cams=cams,
            class_ids=[2, 3],
            positive_class_ids={2, 3},
            background_threshold=0.2,
            foreground_threshold=0.7,
            cam_support_threshold=0.3,
        )

        np.testing.assert_array_equal(
            fused,
            np.array([[255, 2, 0, 3]], dtype=np.uint8),
        )
        self.assertEqual(stats["conflict_pixels"], 1)
        self.assertEqual(stats["cam_foreground_pixels"], 1)
        self.assertEqual(stats["background_pixels"], 1)
        self.assertEqual(stats["sam_foreground_pixels"], 1)

    def test_absent_classes_cannot_be_fused(self) -> None:
        sam = np.full((1, 2), 255, dtype=np.uint8)
        cams = np.array([[[0.9, 0.1]], [[0.1, 0.9]]], dtype=np.float32)

        fused, _stats = fuse_cam_and_sam_with_stats(
            sam_label=sam,
            cams=cams,
            class_ids=[2, 3],
            positive_class_ids={2},
            background_threshold=0.2,
            foreground_threshold=0.7,
        )

        np.testing.assert_array_equal(fused, np.array([[2, 0]], dtype=np.uint8))

    def test_background_only_preserves_sam_and_disables_cam_foreground(self) -> None:
        sam = np.array([[2, 255, 255, 3]], dtype=np.uint8)
        cams = np.array(
            [
                [[0.1, 0.9, 0.0, 0.9]],
                [[0.9, 0.1, 0.0, 0.4]],
            ],
            dtype=np.float32,
        )

        fused, stats = fuse_cam_and_sam_with_stats(
            sam_label=sam,
            cams=cams,
            class_ids=[2, 3],
            positive_class_ids={2, 3},
            background_threshold=0.0,
            foreground_threshold=0.7,
            cam_support_threshold=0.3,
            background_only=True,
        )

        np.testing.assert_array_equal(
            fused,
            np.array([[2, 255, 0, 3]], dtype=np.uint8),
        )
        self.assertEqual(stats["sam_foreground_pixels"], 2)
        self.assertEqual(stats["background_pixels"], 1)
        self.assertEqual(stats["cam_foreground_pixels"], 0)
        self.assertEqual(stats["conflict_pixels"], 0)


class CAMTrainingTests(unittest.TestCase):
    def test_f1_metrics_report_micro_macro_and_per_class(self) -> None:
        micro, macro, per_class = _f1_metrics(
            tp=np.array([2, 0], dtype=np.int64),
            fp=np.array([1, 0], dtype=np.int64),
            fn=np.array([0, 0], dtype=np.int64),
            class_names=("building", "car"),
        )

        self.assertAlmostEqual(micro, 0.8)
        self.assertAlmostEqual(macro, 0.4)
        self.assertEqual(per_class, {"building": 0.8, "car": 0.0})

    def test_cam_validation_rejects_shared_parent_tiles(self) -> None:
        self.assertEqual(
            _parent_image_id("top_potsdam_2_10_x0384_y0768"),
            "top_potsdam_2_10",
        )
        with self.assertRaisesRegex(ValueError, "share parent images"):
            _ensure_parent_disjoint(
                ["top_potsdam_2_10_x0000_y0000"],
                ["top_potsdam_2_10_x0384_y0768"],
            )

        _ensure_parent_disjoint(
            ["top_potsdam_2_10_x0000_y0000"],
            ["top_potsdam_2_11_x0000_y0000"],
        )

    def test_cam_training_output_requires_explicit_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "cam"
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            best_path = checkpoint_dir / "best.pt"
            best_path.write_bytes(b"existing checkpoint")
            log_path = output_dir / "train_log.jsonl"
            log_path.write_text("existing log\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already contains"):
                _prepare_training_output(output_dir)

            self.assertEqual(best_path.read_bytes(), b"existing checkpoint")
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "existing log\n",
            )

            returned_checkpoint_dir, returned_log_path = _prepare_training_output(
                output_dir,
                overwrite=True,
            )
            self.assertEqual(returned_checkpoint_dir, checkpoint_dir)
            self.assertEqual(returned_log_path, log_path)
            self.assertFalse(best_path.exists())
            self.assertEqual(log_path.read_text(encoding="utf-8"), "")


class StudentValidationTests(unittest.TestCase):
    def test_parent_stitching_prefers_patch_centers_without_double_counting(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_root = root / "predictions"
            label_root = root / "labels"
            prediction_root.mkdir()
            label_root.mkdir()
            records = []
            for image_id, x0, values in (
                ("parent_x0000_y0000", 0, np.ones((2, 3), dtype=np.uint8)),
                ("parent_x0001_y0000", 1, np.full((2, 3), 2, dtype=np.uint8)),
            ):
                Image.fromarray(values, mode="L").save(
                    prediction_root / f"{image_id}.png"
                )
                label_path = label_root / f"{image_id}_label.tif"
                tifffile.imwrite(
                    label_path,
                    np.full((2, 3, 3), (255, 0, 0), dtype=np.uint8),
                    photometric="rgb",
                )
                records.append(
                    PatchRecord(
                        image_id=image_id,
                        parent_image_id="parent",
                        label_path=label_path,
                        x0=x0,
                        y0=0,
                        x1=x0 + 3,
                        y1=2,
                    )
                )
            config = parse_config(_config(root))

            prediction, label = stitch_parent(records, prediction_root, config)

            np.testing.assert_array_equal(
                prediction,
                np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8),
            )
            np.testing.assert_array_equal(label, np.zeros((2, 4), dtype=np.uint8))

    def test_segmentation_metrics_report_background_and_foreground_iou(self) -> None:
        metrics = segmentation_metrics(
            np.asarray([[2, 0], [1, 1]], dtype=np.int64),
            ("background", "foreground"),
        )

        self.assertAlmostEqual(metrics["class_iou"]["background"], 2 / 3)
        self.assertAlmostEqual(metrics["class_iou"]["foreground"], 1 / 2)
        self.assertAlmostEqual(metrics["miou"], 7 / 12)
        self.assertAlmostEqual(metrics["foreground_miou"], 1 / 2)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 3 / 4)

    def test_student_validation_rejects_shared_parent_tiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "share parent images"):
            ensure_student_parent_disjoint(
                ["top_potsdam_2_10_x0000_y0000"],
                ["top_potsdam_2_10_x0384_y0768"],
            )

        ensure_student_parent_disjoint(
            ["top_potsdam_2_10_x0000_y0000"],
            ["top_potsdam_2_11_x0000_y0000"],
        )


class PotsdamPatchDatasetTests(unittest.TestCase):
    def test_patch_dataset_is_discoverable_and_has_per_patch_tags(self) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_root = temporary_path / "source"
            image_root = source_root / "4_Ortho_RGBIR"
            label_root = source_root / "5_Labels_all"
            image_root.mkdir(parents=True)
            label_root.mkdir(parents=True)

            image = np.zeros((4, 4, 4), dtype=np.uint8)
            label = np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8)
            label[:2, :2] = (0, 0, 255)
            tifffile.imwrite(
                image_root / "top_potsdam_2_10_RGBIR.tif",
                image,
                photometric="rgb",
            )
            tifffile.imwrite(
                label_root / "top_potsdam_2_10_label.tif",
                label,
                photometric="rgb",
            )

            config_path = temporary_path / "config.json"
            config_path.write_text(
                json.dumps(_config(source_root)),
                encoding="utf-8",
            )
            split_path = temporary_path / "parent_split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "train": ["top_potsdam_2_10"],
                        "val": [],
                        "test": [],
                        "exclude": [],
                    }
                ),
                encoding="utf-8",
            )
            output_root = temporary_path / "patches"
            summary = prepare_patches(
                config_path,
                output_root,
                patch_size=2,
                patch_overlap=0,
                min_class_pixels=1,
                compression="deflate",
                parent_split=split_path,
            )

            self.assertEqual(summary["patches"], 4)
            self.assertEqual(summary["split_parent_images"]["train"], 1)
            self.assertEqual(summary["split_patches"]["train"], 4)
            self.assertEqual(summary["split_patches"]["val"], 0)
            with (output_root / "image_level_labels.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            labels_by_id = {row["image_id"]: row for row in rows}
            self.assertEqual(
                labels_by_id["top_potsdam_2_10_x0000_y0000"]["building"],
                "1",
            )
            self.assertEqual(
                labels_by_id["top_potsdam_2_10_x0002_y0000"]["building"],
                "0",
            )
            with (output_root / "image_level_labels_train.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertEqual(len(train_rows), 4)
            with (output_root / "image_level_labels_val.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with (output_root / "patches.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                metadata_rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in metadata_rows}, {"train"})
            self.assertTrue((output_root / "parent_split.json").exists())

            from sam3_remote_wsss.config import load_config

            patch_config = load_config(output_root / "potsdam_patches_config.json")
            items = discover_potsdam_items(patch_config)
            self.assertEqual(len(items), 4)
            self.assertTrue(all(item.label_path is not None for item in items))
            self.assertEqual(read_rgbir_as_rgb(items[0].image_path, (0, 1, 2)).shape, (2, 2, 3))
            self.assertEqual(patch_config.tile_size, 2)
            self.assertEqual(patch_config.tile_overlap, 0)
            self.assertFalse(patch_config.background_prompting.enabled)

            cam_dataset = PotsdamImageLevelDataset(
                config=patch_config,
                labels_csv=output_root / "image_level_labels.csv",
                image_size=2,
                augment=False,
            )
            self.assertEqual(len(cam_dataset), 4)
            self.assertEqual(tuple(cam_dataset[0]["image"].shape), (3, 2, 2))
            self.assertEqual(float(cam_dataset[0]["target"][1]), 1.0)

            gt_dataset = PotsdamGroundTruthSegDataset(
                config=patch_config,
                labels_csv=output_root / "image_level_labels_train.csv",
                image_size=2,
            )
            self.assertEqual(len(gt_dataset), 4)
            self.assertEqual(tuple(gt_dataset[0]["image"].shape), (3, 2, 2))
            self.assertEqual(tuple(gt_dataset[0]["label"].shape), (2, 2))
            self.assertEqual(
                set(gt_dataset[0]["label"].numpy().reshape(-1).tolist()),
                {2},
            )

            pseudo_root = temporary_path / "pseudo"
            pseudo_root.mkdir()
            for item in items:
                gt_ids = label_rgb_to_ids(
                    read_label_rgb(item.label_path),  # type: ignore[arg-type]
                    patch_config.classes,
                    patch_config.ignore_index,
                    background_colors=patch_config.background_colors,
                )
                Image.fromarray(gt_ids, mode="L").save(pseudo_root / f"{item.image_id}.png")

            metrics_path = temporary_path / "metrics.json"
            with patch(
                "sys.argv",
                [
                    "evaluate_pseudo_labels",
                    "--config",
                    str(output_root / "potsdam_patches_config.json"),
                    "--pseudo-label-dir",
                    str(pseudo_root),
                    "--output",
                    str(metrics_path),
                ],
            ):
                evaluate_main()
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["class_iou"]["background"], 1.0)
            self.assertEqual(metrics["class_iou"]["building"], 1.0)
            self.assertEqual(metrics["input_pseudo_labels"], 4)
            self.assertEqual(metrics["evaluated_images"], 4)
            self.assertEqual(metrics["skipped_images"], 0)
            self.assertEqual(
                metrics["skipped_reasons"],
                {
                    "missing_item": 0,
                    "missing_label": 0,
                    "no_valid_gt": 0,
                },
            )

    def test_parent_split_requires_every_parent_exactly_once(self) -> None:
        with TemporaryDirectory() as temporary:
            split_path = Path(temporary) / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "train": ["top_potsdam_2_10"],
                        "val": ["top_potsdam_2_10"],
                        "test": [],
                        "exclude": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "multiple split groups"):
                load_parent_split(split_path, {"top_potsdam_2_10"})

            split_path.write_text(
                json.dumps(
                    {
                        "train": ["top_potsdam_2_10"],
                        "val": [],
                        "test": [],
                        "exclude": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unassigned"):
                load_parent_split(
                    split_path,
                    {"top_potsdam_2_10", "top_potsdam_2_11"},
                )


def _config(dataset_root: Path) -> dict:
    return {
        "dataset_root": str(dataset_root),
        "image_dir": "4_Ortho_RGBIR",
        "label_dir": "5_Labels_all",
        "sam3_repo": str(dataset_root / "sam3"),
        "checkpoint_path": str(dataset_root / "sam3.pt"),
        "device": "cuda",
        "tile_size": 512,
        "tile_overlap": 128,
        "score_threshold": 0.55,
        "mask_threshold": 0.5,
        "min_mask_area": 32,
        "max_mask_area_ratio": 0.95,
        "conflict_margin": 0.03,
        "ignore_index": 255,
        "rgb_band_indices": [0, 1, 2],
        "prompting": {
            "style": "manual",
            "include_manual_prompts": True,
            "max_prompts_per_class": 1,
        },
        "classes": [
            {
                "id": spec.id,
                "name": spec.name,
                "label_color": list(spec.label_color),
                "prompts": list(spec.prompts),
            }
            for spec in CLASSES
        ],
        "background_colors": [[255, 0, 0]],
        "remoteclip": {"enabled": False},
    }


if __name__ == "__main__":
    unittest.main()
