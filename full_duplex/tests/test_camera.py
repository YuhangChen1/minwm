from __future__ import annotations

import unittest

import torch

from full_duplex.camera import (
    camera_loss,
    camera_to_viewmats_and_Ks,
    viewmats_and_Ks_to_camera,
)


class CameraRepresentationTest(unittest.TestCase):
    def test_camera_roundtrip_and_loss(self) -> None:
        viewmats = torch.eye(4).repeat(3, 1, 1)
        viewmats[:, :3, 3] = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0]])
        Ks = torch.eye(3).repeat(3, 1, 1)
        Ks[:, 0, 0] = 0.505
        Ks[:, 1, 1] = 0.898
        Ks[:, 0, 2] = 0.5
        Ks[:, 1, 2] = 0.5
        camera = viewmats_and_Ks_to_camera(viewmats, Ks)
        reconstructed_viewmats, reconstructed_Ks = camera_to_viewmats_and_Ks(camera)
        torch.testing.assert_close(reconstructed_viewmats, viewmats)
        torch.testing.assert_close(reconstructed_Ks, Ks)
        losses = camera_loss(camera, camera, 1.0, 1.0, 1.0)
        self.assertEqual(losses.total.item(), 0.0)
        self.assertEqual(losses.translation.item(), 0.0)
        self.assertEqual(losses.rotation.item(), 0.0)
        self.assertEqual(losses.intrinsics.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
