from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


CAMERA_REPRESENTATION = "w2c_translation_rotation6d_intrinsics"
CAMERA_DIM = 13


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Encode a rotation matrix with its first two rows, following Zhou et al."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {tuple(matrix.shape)}")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Differentiably map the continuous 6D representation to SO(3)."""
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected (...,6), got {tuple(d6.shape)}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def viewmats_and_Ks_to_camera(viewmats: torch.Tensor, Ks: torch.Tensor) -> torch.Tensor:
    if viewmats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected w2c (...,4,4), got {tuple(viewmats.shape)}")
    if Ks.shape[-2:] != (3, 3) or Ks.shape[:-2] != viewmats.shape[:-2]:
        raise ValueError("Ks must match viewmats leading dimensions")
    translation = viewmats[..., :3, 3]
    rotation = matrix_to_rotation_6d(viewmats[..., :3, :3])
    intrinsics = torch.stack(
        (Ks[..., 0, 0], Ks[..., 1, 1], Ks[..., 0, 2], Ks[..., 1, 2]), dim=-1
    )
    return torch.cat((translation, rotation, intrinsics), dim=-1)


def camera_to_viewmats_and_Ks(camera: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if camera.shape[-1] != CAMERA_DIM:
        raise ValueError(f"Expected camera dim {CAMERA_DIM}, got {camera.shape[-1]}")
    viewmats = torch.zeros(*camera.shape[:-1], 4, 4, device=camera.device, dtype=camera.dtype)
    viewmats[..., :3, :3] = rotation_6d_to_matrix(camera[..., 3:9])
    viewmats[..., :3, 3] = camera[..., :3]
    viewmats[..., 3, 3] = 1
    Ks = torch.zeros(*camera.shape[:-1], 3, 3, device=camera.device, dtype=camera.dtype)
    Ks[..., 0, 0] = camera[..., 9]
    Ks[..., 1, 1] = camera[..., 10]
    Ks[..., 0, 2] = camera[..., 11]
    Ks[..., 1, 2] = camera[..., 12]
    Ks[..., 2, 2] = 1
    return viewmats, Ks


@dataclass
class CameraLosses:
    total: torch.Tensor
    translation: torch.Tensor
    rotation: torch.Tensor
    intrinsics: torch.Tensor


def camera_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_translation: float,
    lambda_rotation: float,
    lambda_intrinsics: float,
) -> CameraLosses:
    translation = F.mse_loss(prediction[..., :3].float(), target[..., :3].float())
    pred_R = rotation_6d_to_matrix(prediction[..., 3:9].float())
    target_R = rotation_6d_to_matrix(target[..., 3:9].float())
    # Chordal SO(3) loss avoids quaternion sign/order ambiguity.
    rotation = F.mse_loss(pred_R, target_R)
    intrinsics = F.mse_loss(prediction[..., 9:].float(), target[..., 9:].float())
    total = (
        lambda_translation * translation
        + lambda_rotation * rotation
        + lambda_intrinsics * intrinsics
    )
    return CameraLosses(total, translation, rotation, intrinsics)
