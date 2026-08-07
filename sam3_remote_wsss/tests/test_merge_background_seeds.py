from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from sam3_remote_wsss.merge_background_seeds import (
    merge_background_seed_dataset,
)


class MergeBackgroundSeedTests(unittest.TestCase):
    def test_only_background_is_copied_and_foreground_wins(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
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
                                "name": "first",
                                "label_color": [255, 255, 255],
                                "prompts": ["first"],
                            },
                            {
                                "id": 2,
                                "name": "second",
                                "label_color": [0, 0, 255],
                                "prompts": ["second"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            labels_csv = root / "labels.csv"
            with labels_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image_id", "first", "second"]
                )
                writer.writeheader()
                writer.writerow({"image_id": "tile", "first": 1, "second": 1})
            foreground_dir = root / "foreground"
            background_dir = root / "background"
            foreground_dir.mkdir()
            background_dir.mkdir()
            Image.fromarray(
                np.asarray([[1, 255], [2, 255]], dtype=np.uint8)
            ).save(foreground_dir / "tile.png")
            Image.fromarray(
                np.asarray([[0, 0], [1, 255]], dtype=np.uint8)
            ).save(background_dir / "tile.png")

            output_dir = root / "output"
            report = merge_background_seed_dataset(
                config_path=config_path,
                labels_csv=labels_csv,
                foreground_pseudo_label_dir=foreground_dir,
                background_seed_label_dir=background_dir,
                output_dir=output_dir,
            )
            merged = np.asarray(Image.open(output_dir / "pseudo_labels/tile.png"))

            np.testing.assert_array_equal(
                merged,
                np.asarray([[1, 0], [2, 255]], dtype=np.uint8),
            )
            self.assertFalse(report["protocol"]["pixel_gt_used"])
            self.assertEqual(report["pixel_totals"]["background_pixels"], 1)
            self.assertEqual(
                report["pixel_totals"]["foreground_background_conflicts"], 1
            )
            self.assertEqual(
                report["pixel_totals"]["discarded_old_foreground_pixels"], 1
            )


if __name__ == "__main__":
    unittest.main()
