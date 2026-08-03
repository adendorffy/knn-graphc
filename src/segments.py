import torch


def segment_bounds_from_durations(durations: torch.Tensor) -> torch.Tensor:
    """Convert per-segment frame counts to inclusive start / exclusive end indices."""
    if durations.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=durations.device)

    ends = durations.cumsum(0)
    starts = ends - durations
    return torch.stack([starts, ends], dim=1)


def segments_to_unit_table(
    durations: torch.Tensor,
    unit_ids: torch.Tensor,
    segment_pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a per-utterance discrete unit table from segment metadata."""
    if segment_pad_mask is not None:
        valid = ~segment_pad_mask
        durations = durations[valid]
        unit_ids = unit_ids[valid]

    bounds = segment_bounds_from_durations(durations)
    return torch.cat([bounds, unit_ids.unsqueeze(-1)], dim=1).to(torch.long)
