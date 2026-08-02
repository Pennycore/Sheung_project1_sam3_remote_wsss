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
from sam3_remote_wsss.fusion import FusionCanvas
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


CLASSES = (
    ClassSpec(1, "impervious_surface", (255, 255, 255), ("impervious surface",)),
    ClassSpec(2, "building", (0, 0, 255), ("building",)),
    ClassSpec(3, "low_vegetation", (0, 255, 255), ("low vegetation",)),
    ClassSpec(4, "tree", (0, 255, 0), ("tree",)),
    ClassSpec(5, "car", (255, 255, 0), ("car",)),
)


class PotsdamMappingTests(unittest.TestCase):
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
