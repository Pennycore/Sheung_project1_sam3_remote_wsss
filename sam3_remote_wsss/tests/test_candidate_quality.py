from __future__ import annotations

import unittest

import numpy as np

from sam3_remote_wsss.analyze_candidate_quality import (
    candidate_quality_record,
    finalize_support_counts,
    merge_support_counts,
    prompt_support_counts,
    summarize_quality_records,
)
from sam3_remote_wsss.candidate_cache import CandidateMask


class CandidateQualityTests(unittest.TestCase):
    def test_candidate_record_reports_semantic_purity_and_contamination(self) -> None:
        gt = np.asarray(
            [
                [2, 2, 0],
                [2, 4, 4],
            ],
            dtype=np.uint8,
        )
        candidate = CandidateMask(
            class_id=2,
            class_name="building",
            prompt="building",
            score=0.8,
            mask=np.asarray(
                [
                    [1, 1, 1],
                    [1, 1, 0],
                ],
                dtype=bool,
            ),
        )

        record = candidate_quality_record(
            image_id="patch_a",
            candidate_index=0,
            candidate=candidate,
            gt=gt,
            num_classes=5,
            id_to_name={0: "background", 2: "building", 4: "tree"},
            ignore_index=255,
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["valid_pixels"], 5)
        self.assertEqual(record["expected_pixels"], 3)
        self.assertEqual(record["background_pixels"], 1)
        self.assertEqual(record["other_foreground_pixels"], 1)
        self.assertAlmostEqual(record["expected_purity"], 0.6)
        self.assertEqual(record["dominant_class_name"], "building")
        self.assertTrue(record["expected_is_dominant"])

        summary = summarize_quality_records([record])
        self.assertAlmostEqual(summary["pixel_weighted_purity"], 0.6)
        self.assertAlmostEqual(summary["background_contamination"], 0.2)
        self.assertAlmostEqual(summary["other_foreground_contamination"], 0.2)

    def test_prompt_support_thresholds_report_precision_recall_and_iou(self) -> None:
        gt = np.asarray(
            [
                [2, 2, 0],
                [2, 4, 4],
            ],
            dtype=np.uint8,
        )
        support = np.asarray(
            [
                [2, 1, 0],
                [0, 2, 1],
            ],
            dtype=np.uint8,
        )
        counts = prompt_support_counts(
            support=support,
            gt=gt,
            class_id=2,
            prompt_count=2,
            ignore_index=255,
        )
        merged = {
            "positive_images": 0,
            "gt_pixels": 0,
            "thresholds": {},
            "exact_support": {},
        }
        merge_support_counts(merged, counts)
        result = finalize_support_counts(merged)

        threshold1 = result["thresholds"]["1"]
        self.assertEqual(threshold1["true_positive"], 2)
        self.assertEqual(threshold1["false_positive"], 2)
        self.assertAlmostEqual(threshold1["precision"], 0.5)
        self.assertAlmostEqual(threshold1["recall"], 2 / 3)
        self.assertAlmostEqual(threshold1["binary_iou"], 0.4)

        threshold2 = result["thresholds"]["2"]
        self.assertEqual(threshold2["true_positive"], 1)
        self.assertEqual(threshold2["false_positive"], 1)
        self.assertAlmostEqual(threshold2["precision"], 0.5)
        self.assertAlmostEqual(threshold2["recall"], 1 / 3)
        self.assertAlmostEqual(threshold2["binary_iou"], 0.25)


if __name__ == "__main__":
    unittest.main()
