import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Wan21"))


class CameraConstructionTest(unittest.TestCase):
    def test_smallest_sample_uses_training_trajectory_semantics(self) -> None:
        from Wan21.scripts.data_preprocessing.build_worldplaygen_lmdb import poses_from_pose_str
        from wan_utils.dataset import build_viewmats_and_Ks

        intrinsics, poses = poses_from_pose_str("right-8, a-11")
        viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
        self.assertEqual(viewmats.shape, (20, 4, 4))
        self.assertEqual(Ks.shape, (20, 3, 3))
        np.testing.assert_allclose(viewmats[0], np.eye(4), atol=1e-7)
        np.testing.assert_allclose(Ks, np.repeat(Ks[:1], 20, axis=0), atol=0)
        self.assertFalse(np.allclose(viewmats[8, :3, :3], np.eye(3)))
        translation_steps = np.linalg.norm(np.diff(viewmats[8:, :3, 3], axis=0), axis=1)
        self.assertTrue(np.all(translation_steps > 0))


if __name__ == "__main__":
    unittest.main()
