#!/usr/bin/env python3
"""Measure reconstruction and planted-feature recovery for one SAE fit."""

import argparse
import json

import torch

from transformer_sae import features_to_tokens, fit_sae, sample_feature_vectors


def main() -> None:
    """Fit an SAE and report how well it recovers the causal concept.

I explain the unit effect euation in explain.md

    Command-line args:
        --seed: Seed for the model and SAE fit (default 777).
        --causal-concept: Index of the planted causal concept (default 2).
        --examples: Number of training examples for the SAE (default 1024).
        --steps: Number of SAE training steps (default 50).

    Prints a JSON object to stdout with:
        explained_variance: Fraction of activation variance reconstructed.
        relative_reconstruction_rmse: Reconstruction RMSE relative to scale.
        feature_count: Number of SAE units.
        active_codes_per_sample: Mean number of nonzero codes per input.
        top_causal_feature_id: Unit with the largest causal effect.
        top_feature_effect: That unit's effect on the readout.
        sum_of_feature_effects: Total effect summed over all units.
        top_feature_effect_share: Top unit's effect divided by the total
            (0.0 if the total is zero).
        causal_feature_activation_correlation: Correlation between the top
            unit's activation and the causal concept.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--causal-concept", type=int, default=2)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    fit = fit_sae(
        args.seed,
        args.causal_concept,
        gain=2.0,
        training_examples=args.examples,
        steps=args.steps,
    )
    with torch.no_grad():
        vectors = sample_feature_vectors(
            args.seed ^ 0xABCDEF, 1024, args.causal_concept
        )
        activations = fit.model.activations(features_to_tokens(vectors))
        codes = fit.sae.encode(activations)
        active = vectors[:, args.causal_concept] > 0
        inactive = ~active
        activation_difference = codes[active].mean(dim=0) - codes[inactive].mean(dim=0)
        feature_effects = activation_difference * (fit.model.readout @ fit.sae.decoder)
        feature_id = int(feature_effects.argmax())
        effect = float(feature_effects[feature_id])
        total_effect = float(feature_effects.sum())
        correlation = torch.corrcoef(
            torch.stack((codes[:, feature_id], vectors[:, args.causal_concept]))
        )[0, 1]
    print(
        json.dumps(
            {
                "explained_variance": fit.explained_variance,
                "relative_reconstruction_rmse": fit.relative_reconstruction_rmse,
                "feature_count": codes.shape[1],
                "active_codes_per_sample": float((codes > 0).sum(dim=1).float().mean()),
                "top_causal_feature_id": feature_id,
                "top_feature_effect": effect,
                "sum_of_feature_effects": total_effect,
                "top_feature_effect_share": effect / total_effect if total_effect else 0.0,
                "causal_feature_activation_correlation": float(correlation),
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()