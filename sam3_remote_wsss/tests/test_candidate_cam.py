from __future__ import annotations

import unittest

import numpy as np

from sam3_remote_wsss.analyze_candidate_cam import (
    score_candidate_cams,
    summarize_cam_records,
)
from sam3_remote_wsss.candidate_cache import CandidateMask


class CandidateCamTests(unittest.TestCase):
    def test_scores_mean_and_top20_inside_candidate_mask(self) -> None:
        candidate = CandidateMask(
            class_id=2,
            class_name="building",
            prompt="building",
            score=0.8,
            mask=np.asarray([[1, 1], [1, 1]], dtype=bool),
            x0=1,
            y0=1,
        )
        cams = np.zeros((2, 4, 4), dtype=np.float32)
        cams[0, 1:3, 1:3] = 0.2
        cams[1, 1:3, 1:3] = np.asarray([[0.1, 0.4], [0.5, 0.6]])

        result = score_candidate_cams(
            candidate=candidate,
            cams=cams,
            cam_class_ids=np.asarray([1, 2]),
            active_class_ids=[1, 2],
        )

        self.assertEqual(result["mean"]["predicted_class_id"], 2)
        self.assertTrue(result["mean"]["agrees_with_candidate"])
        self.assertAlmostEqual(result["mean"]["expected_score"], 0.4)
        self.assertAlmostEqual(result["mean"]["expected_margin"], 0.2)
        self.assertAlmostEqual(result["top20"]["expected_score"], 0.6)

    def test_summary_reports_rejector_precision_and_recall(self) -> None:
        records = [
            self._record(expected_is_dominant=True, predicted=2, purity=0.9),
            self._record(expected_is_dominant=True, predicted=1, purity=0.8),
            self._record(
                expected_is_dominant=False,
                predicted=1,
                purity=0.2,
                dominant_class_id=1,
            ),
        ]

        summary = summarize_cam_records(records, "mean")
        filtered = summary["agreement_filter"]

        self.assertAlmostEqual(summary["agreement_rate"], 1 / 3)
        self.assertAlmostEqual(summary["wrong_candidate_correction_rate"], 1.0)
        self.assertAlmostEqual(filtered["dominant_match_rate_before"], 2 / 3)
        self.assertAlmostEqual(filtered["dominant_match_rate_after"], 1.0)
        self.assertAlmostEqual(filtered["correct_candidate_recall"], 0.5)
        self.assertAlmostEqual(filtered["wrong_candidate_rejection_rate"], 1.0)

    @staticmethod
    def _record(
        expected_is_dominant: bool,
        predicted: int,
        purity: float,
        dominant_class_id: int = 2,
    ) -> dict:
        return {
            "class_id": 2,
            "class_name": "building",
            "expected_is_dominant": expected_is_dominant,
            "dominant_class_id": dominant_class_id,
            "expected_purity": purity,
            "valid_pixels": 10,
            "expected_pixels": round(10 * purity),
            "cam": {
                "mean": {
                    "predicted_class_id": predicted,
                    "expected_margin": 0.1 if predicted == 2 else -0.1,
                    "agrees_with_candidate": predicted == 2,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
