from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

from sam3_remote_wsss.analyze_candidate_recoverability import (
    analyze_candidate_recoverability,
    candidate_coverage_counts,
    finalize_coverage,
    oracle_assignments,
    summarize_reconciliation_audit,
)
from sam3_remote_wsss.candidate_cache import CandidateMask, save_candidate_cache


class CandidateRecoverabilityTests(unittest.TestCase):
    def test_reconciliation_audit_separates_helpful_and_harmful_actions(self) -> None:
        records = [
            self._audit_record(
                source_id=1,
                dominant_id=2,
                assigned_id=2,
                action="relabel",
            ),
            self._audit_record(
                source_id=1,
                dominant_id=1,
                assigned_id=2,
                action="relabel",
            ),
            self._audit_record(
                source_id=1,
                dominant_id=1,
                assigned_id=None,
                action="ignore",
            ),
        ]
        audit = summarize_reconciliation_audit(
            records,
            {0: "background", 1: "surface", 2: "building"},
        )["overall"]

        self.assertEqual(audit["beneficial_relabel_rate"], 0.5)
        self.assertEqual(audit["destructive_relabel_rate"], 0.5)
        self.assertEqual(audit["ignored_source_dominant_rate"], 1.0)

    def test_oracle_relabel_recovers_semantically_wrong_geometry(self) -> None:
        candidates = [
            self._candidate(class_id=1, class_name="surface", right=False),
            self._candidate(class_id=1, class_name="surface", right=True),
        ]
        quality = {
            0: {"dominant_class_id": 1},
            1: {"dominant_class_id": 2},
        }

        assignments, actions = oracle_assignments(
            candidates,
            quality,
            active_class_ids={1, 2},
        )

        self.assertEqual(assignments["baseline"], [1, 1])
        self.assertEqual(assignments["oracle_reject"], [1, None])
        self.assertEqual(assignments["oracle_relabel"], [1, 2])
        self.assertEqual(actions["oracle_relabel"]["relabeled"], 1)

        gt = np.asarray(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.uint8,
        )
        coverage = finalize_coverage(
            candidate_coverage_counts(candidates, gt, [1, 2]),
            {1: "surface", 2: "building"},
        )
        self.assertEqual(
            coverage["per_class"]["building"]["geometric_recall"],
            1.0,
        )
        self.assertEqual(
            coverage["per_class"]["building"]["semantic_recall"],
            0.0,
        )
        self.assertEqual(
            coverage["per_class"]["building"]["recoverable_semantic_gap"],
            1.0,
        )

    def test_end_to_end_report_compares_oracles_and_cam_recovery(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            image_root = dataset_root / "images"
            label_root = dataset_root / "labels"
            image_root.mkdir(parents=True)
            label_root.mkdir(parents=True)
            image_id = "top_potsdam_2_10_x0000_y0000"

            image = np.zeros((4, 4, 4), dtype=np.uint8)
            label = np.zeros((4, 4, 3), dtype=np.uint8)
            label[:, :2] = (255, 255, 255)
            label[:, 2:] = (0, 0, 255)
            tifffile.imwrite(
                image_root / f"{image_id}_RGBIR.tif",
                image,
                photometric="rgb",
            )
            tifffile.imwrite(
                label_root / f"{image_id}_label.tif",
                label,
                photometric="rgb",
            )

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "image_dir": "images",
                        "label_dir": "labels",
                        "sam3_repo": str(root),
                        "checkpoint_path": None,
                        "device": "cpu",
                        "tile_size": 4,
                        "tile_overlap": 0,
                        "score_threshold": 0.5,
                        "mask_threshold": 0.5,
                        "min_mask_area": 1,
                        "max_mask_area_ratio": 1.0,
                        "conflict_margin": 0.03,
                        "ignore_index": 255,
                        "uncovered_label": 255,
                        "rgb_band_indices": [0, 1, 2],
                        "background_colors": [[255, 0, 0]],
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
                    handle,
                    fieldnames=["image_id", "surface", "building"],
                )
                writer.writeheader()
                writer.writerow(
                    {"image_id": image_id, "surface": 1, "building": 1}
                )

            candidate_dir = root / "candidates"
            candidates = [
                self._candidate(class_id=1, class_name="surface", right=False),
                self._candidate(class_id=1, class_name="surface", right=True),
            ]
            save_candidate_cache(
                candidate_dir,
                image_id,
                image_shape=(4, 4),
                candidates=candidates,
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
                class_ids=np.asarray([1, 2], dtype=np.int64),
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "protocol": {"pixel_gt_used": False},
                        "cam_method": "mean",
                        "per_source": {
                            "surface": {
                                "class_id": 1,
                                "threshold": 0.5,
                                "threshold_source": "test",
                            },
                            "building": {
                                "class_id": 2,
                                "threshold": 0.5,
                                "threshold_source": "test",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = analyze_candidate_recoverability(
                config_path=config_path,
                labels_csv=labels_csv,
                candidate_dir=candidate_dir,
                cam_dir=cam_dir,
                calibration_path=calibration_path,
            )

            self.assertEqual(report["evaluated_images"], 1)
            self.assertAlmostEqual(
                report["policies"]["baseline"]["foreground_miou"],
                0.25,
            )
            self.assertAlmostEqual(
                report["policies"]["oracle_reject"]["foreground_miou"],
                0.5,
            )
            self.assertAlmostEqual(
                report["policies"]["oracle_relabel"]["foreground_miou"],
                1.0,
            )
            self.assertAlmostEqual(
                report["policies"]["oracle_relabel"]["oa"],
                1.0,
            )
            self.assertAlmostEqual(
                report["policy_comparisons"]
                ["oracle_relabel_minus_oracle_reject"]
                ["foreground_miou"],
                0.5,
            )
            self.assertEqual(
                report["sam_to_gt_confusion"]["candidate_count_weighted"]
                ["counts"]["surface"]["building"],
                1,
            )
            pair = report["cam_confusion_pair_recoverability"]["mean"][
                "per_pair"
            ]["surface->building"]
            self.assertEqual(pair["candidates"], 1)
            self.assertEqual(pair["cam_correction_rate"], 1.0)
            audit = report["candidate_reconciliation_audit"]["overall"]
            self.assertEqual(audit["actions"], {"keep": 1, "relabel": 1})
            self.assertEqual(audit["beneficial_relabel_rate"], 1.0)
            self.assertEqual(audit["destructive_relabel_rate"], 0.0)

    @staticmethod
    def _candidate(
        class_id: int,
        class_name: str,
        right: bool,
    ) -> CandidateMask:
        mask = np.zeros((4, 4), dtype=bool)
        if right:
            mask[:, 2:] = True
        else:
            mask[:, :2] = True
        return CandidateMask(
            class_id=class_id,
            class_name=class_name,
            prompt=class_name,
            score=0.8,
            mask=mask,
        )

    @staticmethod
    def _audit_record(
        source_id: int,
        dominant_id: int,
        assigned_id: int | None,
        action: str,
    ) -> dict:
        source_name = "surface" if source_id == 1 else "building"
        dominant_name = "surface" if dominant_id == 1 else "building"
        return {
            "source_class_id": source_id,
            "source_class_name": source_name,
            "assigned_class_id": assigned_id,
            "action": action,
            "dominant_class_id": dominant_id,
            "expected_is_dominant": source_id == dominant_id,
            "expected_pixels": 8 if source_id == dominant_id else 2,
            "valid_pixels": 10,
            "gt_distribution": {
                source_name: 8 if source_id == dominant_id else 2,
                dominant_name: 8,
            },
            "margin": 0.8 if action == "relabel" else 0.1,
        }


if __name__ == "__main__":
    unittest.main()
