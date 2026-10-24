#!/usr/bin/env python3
"""Train the fixed SAE weights baked into the task image."""

import argparse
import json
from pathlib import Path

import torch

from transformer_sae import fit_sae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    fit = fit_sae(seed=0x51AE, causal_concept=0, gain=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fit.sae.state_dict(), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "explained_variance": fit.explained_variance,
                "relative_reconstruction_rmse": fit.relative_reconstruction_rmse,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
