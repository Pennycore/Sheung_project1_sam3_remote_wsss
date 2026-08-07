from __future__ import annotations

import unittest

import numpy as np

from sam3_remote_wsss.candidate_cache import CandidateMask
from sam3_remote_wsss.rebuild_candidate_pseudo_labels import (
    cam_keep_decisions,
    fuse_cached_candidates,
)


class CandidateRebuildTests(unittest.TestCase):
    def test_fusion_rebuild_preserves_candidate_scores_and_conflicts(self) -> None:
        candidates = [
            CandidateMask(
                class_id=1,
                class_name="surface",
                prompt="surface",
                score=0.8,
                mask=np.asarray([[1, 1, 0]], dtype=bool),
            ),
            CandidateMask(
                class_id=2,
                class_name="building",
                prompt="building",
                score=0.79,
                mask=np.asarray([[0, 1, 1]], dtype=bool),
            ),
        ]

        label = fuse_cached_candidates(
            image_shape=(1, 3),
            candidates=candidates,
            keep=[True, True],
            ignore_index=255,
            uncovered_label=255,
            conflict_margin=0.03,
        )

        np.testing.assert_array_equal(
            label,
            np.asarray([[1, 255, 2]], dtype=np.uint8),
        )

    def test_cam_rejects_only_configured_classes(self) -> None:
        candidates = [
            CandidateMask(
                class_id=1,
                class_name="impervious_surface",
                prompt="surface",
                score=0.8,
                mask=np.ones((2, 2), dtype=bool),
            ),
            CandidateMask(
                class_id=2,
                class_name="building",
                prompt="building",
                score=0.8,
                mask=np.ones((2, 2), dtype=bool),
            ),
        ]
        cams = np.asarray(
            [
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.9, 0.9], [0.9, 0.9]],
            ],
            dtype=np.float32,
        )

        keep, counts = cam_keep_decisions(
            candidates=candidates,
            cams=cams,
            cam_class_ids=np.asarray([1, 2]),
            active_class_ids=[1, 2],
            reject_classes={"impervious_surface"},
            cam_method="mean",
        )

        self.assertEqual(keep, [False, True])
        self.assertEqual(counts["impervious_surface"]["rejected"], 1)
        self.assertEqual(counts["building"]["checked"], 0)


if __name__ == "__main__":
    unittest.main()
