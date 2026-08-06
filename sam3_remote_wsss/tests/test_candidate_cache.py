from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sam3_remote_wsss.candidate_cache import (
    CandidateMask,
    candidate_cache_exists,
    load_candidate_cache,
    save_candidate_cache,
)
from sam3_remote_wsss.generate_pseudo_labels import _item_output_complete
from sam3_remote_wsss.summarize_candidates import summarize_candidate_dir


class CandidateCacheTests(unittest.TestCase):
    def test_round_trip_preserves_variable_masks_and_prompt_identity(self) -> None:
        first_mask = np.asarray(
            [
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
        second_mask = np.asarray(
            [
                [0, 1],
                [1, 1],
                [0, 0],
            ],
            dtype=bool,
        )

        with TemporaryDirectory() as temporary:
            metadata = save_candidate_cache(
                cache_dir=temporary,
                image_id="patch_a",
                image_shape=(8, 9),
                candidates=[
                    CandidateMask(
                        class_id=2,
                        class_name="building",
                        prompt="overhead view of buildings",
                        score=0.875,
                        mask=first_mask,
                        x0=1,
                        y0=2,
                    ),
                    CandidateMask(
                        class_id=4,
                        class_name="tree",
                        prompt="trees in aerial imagery",
                        score=0.625,
                        mask=second_mask,
                        x0=6,
                        y0=1,
                    ),
                ],
            )

            loaded_metadata, loaded = load_candidate_cache(temporary, "patch_a")

            self.assertTrue(candidate_cache_exists(temporary, "patch_a"))
            self.assertEqual(metadata, loaded_metadata)
            self.assertEqual(loaded_metadata["candidate_count"], 2)
            self.assertEqual(
                loaded_metadata["class_candidate_counts"],
                {"building": 1, "tree": 1},
            )
            self.assertEqual(loaded[0].class_id, 2)
            self.assertEqual(loaded[0].prompt, "overhead view of buildings")
            self.assertAlmostEqual(loaded[0].score, 0.875)
            self.assertEqual((loaded[0].x0, loaded[0].y0), (1, 2))
            np.testing.assert_array_equal(loaded[0].mask, first_mask)
            np.testing.assert_array_equal(loaded[1].mask, second_mask)

    def test_invalid_candidate_bounds_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "bounds exceed"):
                save_candidate_cache(
                    cache_dir=temporary,
                    image_id="patch_a",
                    image_shape=(4, 4),
                    candidates=[
                        CandidateMask(
                            class_id=2,
                            class_name="building",
                            prompt="building",
                            score=0.9,
                            mask=np.ones((3, 3), dtype=bool),
                            x0=2,
                            y0=2,
                        )
                    ],
                )

    def test_skip_existing_requires_cache_only_when_requested(self) -> None:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            pseudo_label = output_dir / "pseudo_labels" / "patch_a.png"
            pseudo_label.parent.mkdir(parents=True)
            pseudo_label.touch()

            self.assertTrue(
                _item_output_complete(output_dir, "patch_a", save_candidates=False)
            )
            self.assertFalse(
                _item_output_complete(output_dir, "patch_a", save_candidates=True)
            )

            save_candidate_cache(
                cache_dir=output_dir / "candidates",
                image_id="patch_a",
                image_shape=(2, 2),
                candidates=[],
            )
            self.assertTrue(
                _item_output_complete(output_dir, "patch_a", save_candidates=True)
            )

    def test_summary_aggregates_classes_prompts_and_distributions(self) -> None:
        with TemporaryDirectory() as temporary:
            candidates = [
                CandidateMask(
                    class_id=2,
                    class_name="building",
                    prompt="building",
                    score=0.8,
                    mask=np.asarray([[1, 1], [0, 0]], dtype=bool),
                ),
                CandidateMask(
                    class_id=2,
                    class_name="building",
                    prompt="rooftops",
                    score=0.9,
                    mask=np.asarray([[1, 1], [1, 0]], dtype=bool),
                ),
            ]
            save_candidate_cache(
                cache_dir=temporary,
                image_id="patch_a",
                image_shape=(2, 2),
                candidates=candidates,
            )
            save_candidate_cache(
                cache_dir=temporary,
                image_id="patch_b",
                image_shape=(2, 2),
                candidates=candidates[:1],
            )

            summary = summarize_candidate_dir(temporary)

            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["candidates"], 3)
            self.assertEqual(summary["class_candidate_counts"], {"building": 3})
            self.assertEqual(summary["class_image_counts"], {"building": 2})
            self.assertEqual(
                summary["prompt_candidate_counts"],
                {"building": {"building": 2, "rooftops": 1}},
            )
            self.assertAlmostEqual(summary["score_percentiles"]["100"], 0.9)
            self.assertEqual(summary["area_percentiles"]["0"], 2.0)


if __name__ == "__main__":
    unittest.main()
