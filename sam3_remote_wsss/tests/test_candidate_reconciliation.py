from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from sam3_remote_wsss.calibrate_candidate_reconciliation import (
    calibrate_candidate_reconciliation,
    fit_two_component_gmm,
)
from sam3_remote_wsss.candidate_cache import CandidateMask, save_candidate_cache
from sam3_remote_wsss.reconcile_candidate_pseudo_labels import (
    reconcile_candidate_pseudo_labels,
)


class CandidateReconciliationTests(unittest.TestCase):
    def test_two_component_gmm_separates_low_and_high_margins(self) -> None:
        fit = fit_two_component_gmm(
            np.asarray([0.08, 0.10, 0.12, 0.78, 0.82, 0.86])
        )
        self.assertIsNotNone(fit)
        self.assertLess(fit["means"][0], 0.2)
        self.assertGreater(fit["means"][1], 0.7)
        self.assertGreater(fit["posterior_boundary"], 0.12)
        self.assertLess(fit["posterior_boundary"], 0.78)

    def test_calibration_and_reconciliation_do_not_require_pixel_gt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(root / "dataset_without_gt"),
                        "image_dir": "images",
                        "label_dir": "labels",
                        "sam3_repo": str(root),
                        "checkpoint_path": None,
                        "device": "cpu",
                        "tile_size": 5,
                        "tile_overlap": 0,
                        "score_threshold": 0.5,
                        "mask_threshold": 0.5,
                        "min_mask_area": 1,
                        "max_mask_area_ratio": 1.0,
                        "conflict_margin": 0.03,
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
            image_id = "top_potsdam_2_10_x0000_y0000"
            labels_csv = root / "labels.csv"
            with labels_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_id", "surface", "building"],
                )
                writer.writeheader()
                writer.writerow(
                    {"image_id": image_id, "surface": 1, "building": 1}
                )

            candidate_dir = root / "candidates"
            candidates = [
                CandidateMask(
                    class_id=1,
                    class_name="surface",
                    prompt="surface",
                    score=0.8,
                    mask=np.ones((1, 1), dtype=bool),
                    x0=index,
                    y0=0,
                )
                for index in range(5)
            ]
            save_candidate_cache(
                candidate_dir,
                image_id,
                image_shape=(1, 5),
                candidates=candidates,
            )

            cam_dir = root / "cams"
            cam_dir.mkdir()
            cams = np.asarray(
                [
                    [[0.90, 0.45, 0.40, 0.10, 0.05]],
                    [[0.10, 0.55, 0.60, 0.90, 0.95]],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                cam_dir / f"{image_id}.npz",
                cams=cams,
                class_ids=np.asarray([1, 2], dtype=np.int64),
            )

            calibration = calibrate_candidate_reconciliation(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                cam_dir=cam_dir,
                min_class_samples=2,
                min_separation=1.0,
            )
            self.assertFalse(calibration["protocol"]["pixel_gt_used"])
            threshold = calibration["per_source"]["surface"]["threshold"]
            self.assertGreater(threshold, 0.2)
            self.assertLess(threshold, 0.8)

            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(calibration, indent=2),
                encoding="utf-8",
            )
            output_dir = root / "reconciled"
            report = reconcile_candidate_pseudo_labels(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                cam_dir=cam_dir,
                calibration_path=calibration_path,
                output_dir=output_dir,
            )

            prediction = np.asarray(
                Image.open(output_dir / "pseudo_labels" / f"{image_id}.png")
            )
            np.testing.assert_array_equal(
                prediction,
                np.asarray([[1, 255, 255, 2, 2]], dtype=np.uint8),
            )
            self.assertEqual(report["candidate_actions"]["kept"], 1)
            self.assertEqual(report["candidate_actions"]["relabeled"], 2)
            self.assertEqual(report["candidate_actions"]["ignored"], 2)
            self.assertFalse(report["protocol"]["pixel_gt_used"])


if __name__ == "__main__":
    unittest.main()
