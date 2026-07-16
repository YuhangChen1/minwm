from __future__ import annotations

import torch


def explicit_rope_apply(
    tensor: torch.Tensor,
    coordinates: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan 3D RoPE using explicit (time, height, width) per token."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,L,H,D], got {tuple(tensor.shape)}")
    if coordinates.shape != (tensor.shape[1], 3):
        raise ValueError(
            f"Coordinates must be [{tensor.shape[1]},3], got {tuple(coordinates.shape)}"
        )
    if tensor.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")

    coordinates = coordinates.to(device=tensor.device, dtype=torch.long)
    frequencies = frequencies.to(tensor.device)
    complex_dim = tensor.shape[-1] // 2
    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    spatial_dim = complex_dim // 3
    freq_t, freq_h, freq_w = frequencies.split(
        [temporal_dim, spatial_dim, spatial_dim], dim=1
    )
    selected = torch.cat(
        (
            freq_t.index_select(0, coordinates[:, 0]),
            freq_h.index_select(0, coordinates[:, 1]),
            freq_w.index_select(0, coordinates[:, 2]),
        ),
        dim=-1,
    ).view(1, tensor.shape[1], 1, complex_dim)

    values = torch.view_as_complex(
        tensor.to(torch.float64).reshape(*tensor.shape[:-1], complex_dim, 2)
    )
    rotated = torch.view_as_real(values * selected).flatten(-2)
    return rotated.to(dtype=tensor.dtype)
