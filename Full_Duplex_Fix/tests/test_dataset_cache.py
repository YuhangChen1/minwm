import json
import tempfile
import unittest
from pathlib import Path

from Full_Duplex_Fix.checkpoint_io import replace_with_link
from Full_Duplex_Fix.dataset_cache import (
    deterministic_split,
    load_aligned_input_manifest,
    normalize_pose_str,
)


class DatasetCacheTest(unittest.TestCase):
    def test_pose_normalization_and_manifest_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "000000_right8a11" / "gen.mp4"
            video.parent.mkdir()
            video.touch()
            manifest = root / "input.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "caption": "caption",
                            "pose_str": "right-8, a-11",
                            "video_path": str(video),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            samples = load_aligned_input_manifest(
                manifest, project_root=root, expected_count=1
            )
            self.assertEqual(normalize_pose_str("right-8, a-11"), "right8a11")
            self.assertEqual(samples[0]["normalized_pose"], "right8a11")

    def test_split_is_complete_disjoint_and_deterministic(self) -> None:
        first = deterministic_split(1000, validation_size=50, seed=17)
        second = deterministic_split(1000, validation_size=50, seed=17)
        self.assertEqual(first, second)
        train, validation = first
        self.assertEqual(len(train), 950)
        self.assertEqual(len(validation), 50)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(1000)))

    def test_checkpoint_alias_does_not_duplicate_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "step_000500.pt"
            source.write_bytes(b"checkpoint")
            alias = replace_with_link(source, root / "latest.pt")
            self.assertEqual(source.stat().st_ino, alias.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
