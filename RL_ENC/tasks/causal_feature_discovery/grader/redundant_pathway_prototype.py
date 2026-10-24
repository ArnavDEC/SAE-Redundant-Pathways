"""Construct two alternate SAE pathways for the same model behavior."""

from copy import deepcopy

import torch

from transformer_sae import SAE_WIDTH, TopKSparseAutoencoder


PRIMARY_PATH_UNIT = SAE_WIDTH - 2
ALTERNATE_PATH_UNIT = SAE_WIDTH - 1
DECOYS_PER_PATH = 12
PRIMARY_DECOY_UNITS = tuple(
    range(SAE_WIDTH - 2 - 2 * DECOYS_PER_PATH, SAE_WIDTH - 2 - DECOYS_PER_PATH)
)
ALTERNATE_DECOY_UNITS = tuple(
    range(SAE_WIDTH - 2 - DECOYS_PER_PATH, SAE_WIDTH - 2)
)
DECOY_DIRECTION_OFFSETS = (*range(1, 6), *range(7, 12), 1, 2)
ENCODER_SCALE = 8.0
DECOY_ENCODER_SCALE = 8.15


def make_redundant_pathway_sae(
    base_sae: TopKSparseAutoencoder,
    concept_directions: torch.Tensor,
    primary_concept: int,
    alternate_concept: int,
) -> TopKSparseAutoencoder:
    """Replace two units with dedicated encoders for parallel sufficient paths.

    Each route has a dedicated causal unit and twelve higher-activation
    decoys. The decoys encode the route inputs but decode into directions
    ignored by the model readout. The saturating OR readout leaves behavior
    intact if the other route remains active. This is a constructed level,
    separate from SAE training.

    Args:
        base_sae: The trained SAE to copy and modify.
        concept_directions: Concept directions, one column per concept.
        primary_concept: Concept index for the primary route.
        alternate_concept: Concept index for the alternate route.

    Returns:
        A new frozen SAE with two redundant causal units and 24 decoys.

    Raises:
        ValueError: If a concept index is out of range or the two are equal.
    """
    concept_count = concept_directions.shape[1]
    if not 0 <= primary_concept < concept_count:
        raise ValueError("primary_concept is outside the concept vocabulary")
    if not 0 <= alternate_concept < concept_count:
        raise ValueError("alternate_concept is outside the concept vocabulary")
    if primary_concept == alternate_concept:
        raise ValueError("primary and alternate concepts must be distinct")

    result = deepcopy(base_sae)
    with torch.no_grad():
        pathway_units = {
            PRIMARY_PATH_UNIT,
            ALTERNATE_PATH_UNIT,
            *PRIMARY_DECOY_UNITS,
            *ALTERNATE_DECOY_UNITS,
        }
        pathway_directions = concept_directions[
            :, [primary_concept, alternate_concept]
        ]
        for unit_id in range(SAE_WIDTH):
            if unit_id not in pathway_units:
                result.decoder[:, unit_id].sub_(
                    pathway_directions
                    @ (pathway_directions.T @ result.decoder[:, unit_id])
                )
        for unit_id, concept_id in (
            (PRIMARY_PATH_UNIT, primary_concept),
            (ALTERNATE_PATH_UNIT, alternate_concept),
        ):
            direction = concept_directions[:, concept_id]
            result.encoder.weight[unit_id].copy_(ENCODER_SCALE * direction)
            result.encoder.bias[unit_id].copy_(
                torch.dot(result.encoder.weight[unit_id], result.mean)
            )
            result.decoder[:, unit_id].copy_(direction)
        decoy_specs = [
            (unit_id, primary_concept, offset)
            for unit_id, offset in zip(
                PRIMARY_DECOY_UNITS, DECOY_DIRECTION_OFFSETS, strict=True
            )
        ] + [
            (unit_id, alternate_concept, offset)
            for unit_id, offset in zip(
                ALTERNATE_DECOY_UNITS, DECOY_DIRECTION_OFFSETS, strict=True
            )
        ]
        for unit_id, input_concept, offset in decoy_specs:
            result.encoder.weight[unit_id].copy_(
                DECOY_ENCODER_SCALE * concept_directions[:, input_concept]
            )
            result.encoder.bias[unit_id].copy_(
                torch.dot(result.encoder.weight[unit_id], result.mean)
            )
            result.decoder[:, unit_id].copy_(
                concept_directions[:, (input_concept + offset) % concept_count]
            )
        result.top_k = 2 + len(PRIMARY_DECOY_UNITS) + len(ALTERNATE_DECOY_UNITS)
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result