#!/usr/bin/env python3
"""Print controlled-condition diagnostics for the shared-unit prototype."""

import argparse
import json
from pathlib import Path

import torch

from private_model import MODEL_ROTATION_SEED
from transformer_sae import (
    MODEL_FEATURES,
    TinyAttentionModel,
    TopKSparseAutoencoder,
    features_to_tokens,
    reference_sae,
)
from polysemantic_prototype import POLYSEMANTIC_UNIT, make_polysemantic_sae


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    if args.artifact is not None:
        base_sae = TopKSparseAutoencoder()
        state = torch.load(args.artifact, map_location="cpu", weights_only=True)
        base_sae.load_state_dict(state)
    else:
        base_sae = reference_sae()

    results = []
    for target in range(MODEL_FEATURES):
        nuisance = (target + 2) % MODEL_FEATURES
        model = TinyAttentionModel(MODEL_ROTATION_SEED, target, gain=2.0)
        sae = make_polysemantic_sae(
            base_sae, model.concept_directions, target, nuisance
        )
        vectors = torch.zeros(4, MODEL_FEATURES)
        vectors[1, target] = 1
        vectors[2, nuisance] = 1
        vectors[3, target] = 1
        vectors[3, nuisance] = 1

        with torch.no_grad():
            activations = model.activations(features_to_tokens(vectors))
            codes = sae.encode(activations)
            logits = model.logits_from_activations(activations)
            unit_activation = codes[:, POLYSEMANTIC_UNIT]
            decoder_projection = model.readout @ sae.decoder[:, POLYSEMANTIC_UNIT]

        results.append({
            "target_concept": target,
            "nuisance_concept": nuisance,
            "unit_activation_neither_target_nuisance_both": [
                round(float(value), 4) for value in unit_activation
            ],
            "largest_other_unit_target_only": round(
                float(
                    torch.cat(
                        (
                            codes[1, :POLYSEMANTIC_UNIT],
                            codes[1, POLYSEMANTIC_UNIT + 1 :],
                        )
                    ).max()
                ),
                4,
            ),
            "largest_other_unit_nuisance_only": round(
                float(
                    torch.cat(
                        (
                            codes[2, :POLYSEMANTIC_UNIT],
                            codes[2, POLYSEMANTIC_UNIT + 1 :],
                        )
                    ).max()
                ),
                4,
            ),
            "unit_target_gap_contribution_without_nuisance": round(
                float((unit_activation[1] - unit_activation[0]) * decoder_projection), 4
            ),
            "unit_target_gap_contribution_with_nuisance": round(
                float((unit_activation[3] - unit_activation[2]) * decoder_projection), 4
            ),
            "model_target_gap_without_nuisance": round(float(logits[1] - logits[0]), 4),
            "model_target_gap_with_nuisance": round(float(logits[3] - logits[2]), 4),
            "model_nuisance_effect_without_target": round(float(logits[2] - logits[0]), 4),
            "shared_unit_selected_for_both": bool(
                codes[1, POLYSEMANTIC_UNIT] > 0 and codes[2, POLYSEMANTIC_UNIT] > 0
            ),
        })
    print(json.dumps({"shared_unit": POLYSEMANTIC_UNIT, "pairs": results}, indent=2))


if __name__ == "__main__":
    main()