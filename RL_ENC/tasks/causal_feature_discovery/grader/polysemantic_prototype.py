"""Construct a deterministic SAE with one shared target/nuisance unit.

This is a research prototype used by the explicitly labeled polysemantic
episode level. It leaves the base episode's SAE unchanged.
"""

from copy import deepcopy

import torch

from transformer_sae import SAE_WIDTH, TopKSparseAutoencoder


POLYSEMANTIC_UNIT = SAE_WIDTH - 1
ENCODER_SCALE = 1.5


def make_polysemantic_sae(
    base_sae: TopKSparseAutoencoder,
    concept_directions: torch.Tensor,
    target_concept: int,
    nuisance_concept: int,
    *,
    unit_id: int = POLYSEMANTIC_UNIT,
    encoder_scale: float = ENCODER_SCALE,
) -> TopKSparseAutoencoder:
    """Return a copy with one encoder unit responding to both concept axes.

    The replacement decoder points equally along the two orthogonal concept
    directions. This guarantees a deliberately shared unit without changing
    the base SAE or its other units.
    """
    if concept_directions.ndim != 2:
        raise ValueError("concept_directions must be a rank-2 tensor")
    concept_count = concept_directions.shape[1]
    if not 0 <= target_concept < concept_count:
        raise ValueError("target_concept is outside the concept vocabulary")
    if not 0 <= nuisance_concept < concept_count:
        raise ValueError("nuisance_concept is outside the concept vocabulary")
    if target_concept == nuisance_concept:
        raise ValueError("target and nuisance concepts must be distinct")
    if not 0 <= unit_id < SAE_WIDTH:
        raise ValueError(f"unit_id must be between 0 and {SAE_WIDTH - 1}")
    if not torch.isfinite(torch.tensor(encoder_scale)) or encoder_scale <= 0:
        raise ValueError("encoder_scale must be finite and positive")

    result = deepcopy(base_sae)
    target_direction = concept_directions[:, target_concept]
    nuisance_direction = concept_directions[:, nuisance_concept]
    with torch.no_grad():
        result.encoder.weight[unit_id].copy_(
            encoder_scale * (target_direction + nuisance_direction)
        )
        result.encoder.bias[unit_id].copy_(
            torch.dot(result.encoder.weight[unit_id], result.mean)
        )
        result.decoder[:, unit_id].copy_(
            0.5 * (target_direction + nuisance_direction)
        )
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result
