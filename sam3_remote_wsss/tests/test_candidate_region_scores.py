from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import tifffile

from sam3_remote_wsss.analyze_candidate_region_semantics import (
    analyze_candidate_region_semantics,
    summarize_assignment_records,
)
from sam3_remote_wsss.candidate_cache import CandidateMask, save_candidate_cache
from sam3_remote_wsss.calibrate_candidate_visual_prototypes import (
    calibrate_candidate_visual_prototypes,
)
from sam3_remote_wsss.candidate_region_scores import (
    active_region_decisions,
    build_class_text_prototypes,
    candidate_cache_fingerprint,
    encode_candidate_regions,
    load_region_score_cache,
    make_candidate_region_views,
    save_region_score_cache,
    score_candidate_regions,
)
from sam3_remote_wsss.config import ClassSpec
from sam3_remote_wsss.reconcile_candidate_visual_prototypes import (
    reconcile_candidate_visual_prototypes,
)
from sam3_remote_wsss.candidate_visual_prototypes import (
    load_visual_prototype_calibration,
    robust_visual_prototype,
    save_visual_prototype_calibration,
)


class FakeEncoder:
    def encode_texts(self, prompts):
        mapping = {
            "a": (1.0, 0.0),
            "b": (0.0, 1.0),
            "c": (0.0, 2.0),
        }
        features = np.asarray([mapping[prompt] for prompt in prompts], dtype=np.float32)
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def encode_images(self, images, batch_size=32):
        del images, batch_size
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class CandidateRegionScoreTests(unittest.TestCase):
    def test_region_views_keep_mask_and_dim_context(self) -> None:
        image = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        candidate = CandidateMask(1, "first", "a", 0.8, mask, x0=3, y0=2)

        views = make_candidate_region_views(
            image,
            candidate,
            context_ratio=0.0,
            min_crop_size=6,
            background_retain=0.25,
        )

        self.assertEqual(views.context.shape, (6, 6, 3))
        self.assertEqual(views.masked.shape, (6, 6, 3))
        self.assertAlmostEqual(views.mask_fraction, 4 / 36)
        self.assertEqual(int((views.masked == 100).all(axis=2).sum()), 4)
        self.assertEqual(int((views.masked == 25).all(axis=2).sum()), 32)

    def test_manual_prompt_features_form_class_prototypes(self) -> None:
        classes = (
            ClassSpec(1, "first", (1, 1, 1), ("a", "b")),
            ClassSpec(2, "second", (2, 2, 2), ("c",)),
        )

        class_ids, prototypes, groups = build_class_text_prototypes(
            FakeEncoder(), classes
        )

        np.testing.assert_array_equal(class_ids, [1, 2])
        np.testing.assert_allclose(
            prototypes,
            [[2**-0.5, 2**-0.5], [0.0, 1.0]],
            atol=1e-6,
        )
        self.assertEqual(groups, {"first": ["a", "b"], "second": ["c"]})

    def test_context_and_masked_features_are_fused(self) -> None:
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        candidate = CandidateMask(
            1,
            "first",
            "a",
            0.8,
            np.ones((2, 2), dtype=bool),
            x0=3,
            y0=3,
        )

        scores, boxes, fractions = score_candidate_regions(
            image,
            [candidate],
            FakeEncoder(),
            np.eye(2, dtype=np.float32),
            min_crop_size=4,
        )

        np.testing.assert_allclose(scores, [[2**-0.5, 2**-0.5]], atol=1e-6)
        self.assertEqual(boxes.shape, (1, 4))
        self.assertEqual(fractions.shape, (1,))

    def test_empty_candidate_features_keep_requested_dimension(self) -> None:
        features, boxes, fractions = encode_candidate_regions(
            image_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            candidates=[],
            encoder=object(),
            feature_dimension=512,
        )

        self.assertEqual(features.shape, (0, 512))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(fractions.shape, (0,))

    def test_empty_candidates_allow_no_active_classes(self) -> None:
        predicted, margins = active_region_decisions(
            scores=np.empty((0, 5), dtype=np.float32),
            class_ids=np.arange(1, 6, dtype=np.int16),
            active_class_ids=[],
        )

        self.assertEqual(predicted.shape, (0,))
        self.assertEqual(margins.shape, (0,))

    def test_region_cache_is_bound_to_candidate_cache_hash(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            candidate_dir = root / "candidates"
            score_dir = root / "scores"
            image_id = "tile"
            candidate = CandidateMask(
                1,
                "first",
                "a",
                0.8,
                np.ones((2, 2), dtype=bool),
            )
            save_candidate_cache(candidate_dir, image_id, (4, 4), [candidate])
            fingerprint = candidate_cache_fingerprint(candidate_dir, image_id)
            save_region_score_cache(
                score_dir,
                image_id,
                scores=np.asarray([[0.8, 0.2]], dtype=np.float32),
                class_ids=np.asarray([1, 2]),
                active_class_ids=[1, 2],
                crop_boxes=np.asarray([[0, 0, 4, 4]]),
                mask_fractions=np.asarray([0.25]),
                candidate_fingerprint=fingerprint,
                metadata={"model_name": "fake"},
            )

            metadata, arrays = load_region_score_cache(
                score_dir, image_id, candidate_dir
            )

            self.assertEqual(metadata["candidate_cache_sha256"], fingerprint)
            np.testing.assert_array_equal(arrays["predicted_class_ids"], [1])
            self.assertAlmostEqual(float(arrays["margins"][0]), 0.6, places=6)

            data_path = candidate_dir / f"{image_id}.npz"
            data_path.write_bytes(data_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed after region scoring"):
                load_region_score_cache(score_dir, image_id, candidate_dir)

    def test_robust_visual_prototype_rejects_feature_outlier(self) -> None:
        features = np.asarray(
            [[1.0, 0.0], [0.99, 0.1], [0.98, -0.1], [-1.0, 0.0]],
            dtype=np.float32,
        )

        prototype, retained, similarities = robust_visual_prototype(
            features, keep_fraction=0.75, iterations=2
        )

        self.assertEqual(retained.size, 3)
        self.assertNotIn(3, retained.tolist())
        self.assertGreater(float(prototype[0]), 0.99)
        self.assertLess(float(similarities[3]), -0.99)

    def test_visual_prototype_calibration_does_not_require_pixel_gt(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_id = "top_potsdam_2_10_x0000_y0000"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(root / "missing_dataset"),
                        "image_dir": "images",
                        "label_dir": "labels",
                        "sam3_repo": str(root),
                        "checkpoint_path": None,
                        "device": "cpu",
                        "ignore_index": 255,
                        "uncovered_label": 255,
                        "classes": [
                            {
                                "id": 1,
                                "name": "surface",
                                "label_color": [255, 255, 255],
                                "prompts": ["surface"],
                            },
                            {
                                "id": 2,
                                "name": "building",
                                "label_color": [0, 0, 255],
                                "prompts": ["building"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            labels_csv = root / "labels.csv"
            with labels_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image_id", "surface", "building"]
                )
                writer.writeheader()
                writer.writerow(
                    {"image_id": image_id, "surface": 1, "building": 1}
                )
            candidate_dir = root / "candidates"
            candidates = [
                CandidateMask(
                    1,
                    "surface",
                    "surface",
                    0.9,
                    np.ones((4, 2), dtype=bool),
                    x0=0,
                ),
                CandidateMask(
                    2,
                    "building",
                    "building",
                    0.9,
                    np.ones((4, 2), dtype=bool),
                    x0=2,
                ),
            ]
            save_candidate_cache(candidate_dir, image_id, (4, 4), candidates)
            score_dir = root / "scores"
            save_region_score_cache(
                score_dir,
                image_id,
                scores=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
                class_ids=np.asarray([1, 2]),
                active_class_ids=[1, 2],
                crop_boxes=np.asarray([[0, 0, 2, 4], [2, 0, 4, 4]]),
                mask_fractions=np.asarray([1.0, 1.0]),
                candidate_fingerprint=candidate_cache_fingerprint(
                    candidate_dir, image_id
                ),
                metadata={"model_name": "fake", "weights_source": "fake.pt"},
                region_features=np.asarray(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
                ),
            )
            cam_dir = root / "cams"
            cam_dir.mkdir()
            cams = np.zeros((2, 4, 4), dtype=np.float32)
            cams[0, :, :2] = 1.0
            cams[1, :, 2:] = 1.0
            np.savez_compressed(
                cam_dir / f"{image_id}.npz",
                cams=cams,
                class_ids=np.asarray([1, 2]),
            )
            output = root / "calibration.json"

            summary = calibrate_candidate_visual_prototypes(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                region_score_dir=score_dir,
                cam_dir=cam_dir,
                output_path=output,
                min_seeds_per_class=1,
            )
            metadata, class_ids, prototypes = load_visual_prototype_calibration(
                output
            )

            self.assertFalse(summary["protocol"]["pixel_gt_used"])
            self.assertEqual(metadata["evaluated_images"], 1)
            np.testing.assert_array_equal(class_ids, [1, 2])
            np.testing.assert_allclose(prototypes, np.eye(2), atol=1e-6)

    def test_assignment_audit_separates_beneficial_and_destructive(self) -> None:
        records = [
            {
                "class_id": 1,
                "class_name": "first",
                "dominant_class_id": 2,
                "valid_pixels": 10,
                "gt_distribution": {"second": 8, "first": 2},
                "assignments": {"consensus": 2},
            },
            {
                "class_id": 1,
                "class_name": "first",
                "dominant_class_id": 1,
                "valid_pixels": 10,
                "gt_distribution": {"first": 9, "second": 1},
                "assignments": {"consensus": 2},
            },
        ]

        summary = summarize_assignment_records(
            records, "consensus", {0: "background", 1: "first", 2: "second"}
        )

        self.assertEqual(summary["relabeled"], 2)
        self.assertEqual(summary["beneficial_relabels"], 1)
        self.assertEqual(summary["destructive_relabels"], 1)
        self.assertAlmostEqual(summary["pixel_weighted_assigned_purity"], 9 / 20)

    def test_end_to_end_consensus_recovers_wrong_candidate_class(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "dataset"
            image_root = dataset / "images"
            label_root = dataset / "labels"
            image_root.mkdir(parents=True)
            label_root.mkdir(parents=True)
            image_id = "top_potsdam_2_10_x0000_y0000"
            tifffile.imwrite(
                image_root / f"{image_id}_RGBIR.tif",
                np.zeros((4, 4, 4), dtype=np.uint8),
                photometric="rgb",
            )
            label = np.zeros((4, 4, 3), dtype=np.uint8)
            label[:, :2] = (255, 255, 255)
            label[:, 2:] = (0, 0, 255)
            tifffile.imwrite(
                label_root / f"{image_id}_label.tif", label, photometric="rgb"
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset),
                        "image_dir": "images",
                        "label_dir": "labels",
                        "sam3_repo": str(root),
                        "checkpoint_path": None,
                        "device": "cpu",
                        "ignore_index": 255,
                        "uncovered_label": 255,
                        "rgb_band_indices": [0, 1, 2],
                        "classes": [
                            {
                                "id": 1,
                                "name": "surface",
                                "label_color": [255, 255, 255],
                                "prompts": ["surface"],
                            },
                            {
                                "id": 2,
                                "name": "building",
                                "label_color": [0, 0, 255],
                                "prompts": ["building"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            labels_csv = root / "labels.csv"
            with labels_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image_id", "surface", "building"]
                )
                writer.writeheader()
                writer.writerow(
                    {"image_id": image_id, "surface": 1, "building": 1}
                )

            candidate_dir = root / "candidates"
            candidates = [
                CandidateMask(
                    1,
                    "surface",
                    "surface",
                    0.9,
                    np.ones((4, 2), dtype=bool),
                    x0=0,
                ),
                CandidateMask(
                    1,
                    "surface",
                    "surface",
                    0.9,
                    np.ones((4, 2), dtype=bool),
                    x0=2,
                ),
            ]
            save_candidate_cache(candidate_dir, image_id, (4, 4), candidates)

            score_dir = root / "scores"
            save_region_score_cache(
                score_dir,
                image_id,
                scores=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
                class_ids=np.asarray([1, 2]),
                active_class_ids=[1, 2],
                crop_boxes=np.asarray([[0, 0, 2, 4], [2, 0, 4, 4]]),
                mask_fractions=np.asarray([1.0, 1.0]),
                candidate_fingerprint=candidate_cache_fingerprint(
                    candidate_dir, image_id
                ),
                metadata={"model_name": "fake", "weights_source": "fake.pt"},
                region_features=np.asarray(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
                ),
            )
            cam_dir = root / "cams"
            cam_dir.mkdir()
            cams = np.zeros((2, 4, 4), dtype=np.float32)
            cams[0, :, :2] = 0.9
            cams[0, :, 2:] = 0.1
            cams[1, :, :2] = 0.1
            cams[1, :, 2:] = 0.9
            np.savez_compressed(
                cam_dir / f"{image_id}.npz",
                cams=cams,
                class_ids=np.asarray([1, 2]),
            )

            report, records = analyze_candidate_region_semantics(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                region_score_dir=score_dir,
                cam_dir=cam_dir,
            )

            self.assertEqual(report["evaluated_images"], 1)
            self.assertEqual(len(records), 2)
            self.assertAlmostEqual(
                report["metrics"]["baseline"]["foreground_miou"], 0.25
            )
            consensus = report["metrics"]["cam_region_consensus"]
            self.assertAlmostEqual(consensus["foreground_miou"], 1.0)
            self.assertAlmostEqual(consensus["foreground_mf1"], 1.0)
            self.assertAlmostEqual(consensus["oa"], 1.0)

            calibration = root / "visual_prototypes.json"
            save_visual_prototype_calibration(
                calibration,
                class_ids=np.asarray([1, 2]),
                prototypes=np.eye(2, dtype=np.float32),
                metadata={
                    "model_name": "fake",
                    "weights_source": "fake.pt",
                    "feature_dimension": 2,
                    "protocol": {"pixel_gt_used": False},
                },
            )
            prototype_report, _ = analyze_candidate_region_semantics(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                region_score_dir=score_dir,
                cam_dir=cam_dir,
                prototype_calibration=calibration,
            )
            self.assertAlmostEqual(
                prototype_report["metrics"]["cam_region_consensus"]["miou"],
                1.0,
            )

            (label_root / f"{image_id}_label.tif").unlink()
            output_dir = root / "reconciled"
            export_report = reconcile_candidate_visual_prototypes(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                region_score_dir=score_dir,
                prototype_calibration=calibration,
                cam_dir=cam_dir,
                output_dir=output_dir,
            )
            exported = np.asarray(
                Image.open(output_dir / "pseudo_labels" / f"{image_id}.png")
            )
            np.testing.assert_array_equal(
                exported,
                np.asarray([[1, 1, 2, 2]] * 4, dtype=np.uint8),
            )
            self.assertFalse(export_report["protocol"]["pixel_gt_used"])
            self.assertEqual(export_report["processed_images"], 1)
            self.assertEqual(
                export_report["candidate_actions"],
                {"keep": 1, "relabel": 1},
            )


if __name__ == "__main__":
    unittest.main()
