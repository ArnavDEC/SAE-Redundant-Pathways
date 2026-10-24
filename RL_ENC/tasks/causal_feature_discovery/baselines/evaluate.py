#!/usr/bin/env python3
"""Evaluate observation-only, random, and oracle baselines on SAE episodes."""

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grader"))
from private_model import Episode, MAX_REPORTED_FEATURES  # noqa: E402
from grader import grade_submission  # noqa: E402

TESTS = ("ablate", "steer", "patch")


def _intervals(
    estimates: tuple[float, float, float],
    half_width: float = 0.15,
) -> list[dict[str, float | str]]:
    return [
        {
            "test": test,
            "lower": estimate - half_width,
            "upper": estimate + half_width,
        }
        for test, estimate in zip(TESTS, estimates, strict=True)
    ]


def _estimated_effects(
    episode: Episode, feature_ids: list[int], slopes: list[float]
) -> tuple[float, float, float]:
    rows = episode.observe("train", 64, nonce=2)["observations"]
    means = [
        statistics.fmean(row["features"][feature_id] for row in rows)
        for feature_id in feature_ids
    ]
    ablate = math.fsum(
        mean * slope for mean, slope in zip(means, slopes, strict=True)
    )
    steer = 0.5 * math.fsum(slopes)
    patch = math.fsum(
        mean * slope for mean, slope in zip(means, slopes, strict=True)
    ) / 2.0
    return ablate, steer, patch


def greedy_submission(episode: Episode, seed: int) -> dict[str, object]:
    del seed
    rows = episode.observe("train", 64, nonce=1)["observations"]
    outputs = [row["logit_diff"] for row in rows]
    candidates: list[tuple[float, int, float, float]] = []
    output_mean = statistics.fmean(outputs)
    for feature_id in range(episode.feature_count):
        values = [row["features"][feature_id] for row in rows]
        mean = statistics.fmean(values)
        variance = statistics.fmean((value - mean) ** 2 for value in values)
        covariance = statistics.fmean(
            (value - mean) * (output - output_mean)
            for value, output in zip(values, outputs, strict=True)
        )
        slope = covariance / variance if variance > 1e-12 else 0.0
        contribution = slope * mean
        candidates.append((abs(contribution), feature_id, contribution, slope))

    selected = sorted(candidates, reverse=True)[:MAX_REPORTED_FEATURES]
    total_mass = math.fsum(abs(item[2]) for item in selected)
    if total_mass > 0:
        components = [
            {"feature_id": feature_id, "share": contribution / total_mass}
            for _, feature_id, contribution, _ in selected
        ]
    else:
        components = []
    feature_ids = [item[1] for item in selected]
    slopes = [item[3] for item in selected]
    estimates = _estimated_effects(episode, feature_ids, slopes)
    return {
        "components": components,
        "predictions": _intervals(estimates),
    }


def random_submission(episode: Episode, seed: int) -> dict[str, object]:
    rng = random.Random(seed ^ 0x52414E444F4D)
    feature_id = rng.randrange(episode.feature_count)
    return {
        "components": [{"feature_id": feature_id, "share": 1.0}],
        "predictions": _intervals((0.0, 0.0, 0.0), 0.05),
    }


def oracle_submission(episode: Episode, seed: int) -> dict[str, object]:
    del seed
    shares = episode.heldout_shares()
    selected = sorted(
        range(episode.feature_count),
        key=lambda feature_id: abs(shares[feature_id]),
        reverse=True,
    )[:MAX_REPORTED_FEATURES]
    components = [
        {"feature_id": feature_id, "share": shares[feature_id]}
        for feature_id in selected
    ]
    estimates = tuple(
        episode.heldout_effect(test, selected) for test in TESTS
    )
    return {
        "components": components,
        "predictions": _intervals(estimates, 0.01),
    }


def hedger_submission(episode: Episode, seed: int) -> dict[str, object]:
    del seed
    shares = episode.heldout_shares()
    selected = sorted(
        range(episode.feature_count),
        key=lambda feature_id: abs(shares[feature_id]),
        reverse=True,
    )[:MAX_REPORTED_FEATURES]
    return {
        "components": [
            {"feature_id": feature_id, "share": shares[feature_id]}
            for feature_id in selected
        ],
        "predictions": _intervals((0.0, 0.0, 0.0), 100.0),
    }


def score_submission(
    episode: Episode, submission: dict[str, object]
) -> tuple[float, float, float]:
    result = grade_submission(submission, episode)
    score = result["score"]
    feedback = result["feedback"]
    if not isinstance(score, (int, float)) or not isinstance(feedback, dict):
        raise RuntimeError(f"Production grader returned an invalid result: {result}")
    return (
        float(score),
        float(feedback["causal_share_recovery"]),
        float(feedback["heldout_prediction"]),
    )


def evaluate(seed_count: int) -> dict[str, tuple[float, float, float]]:
    totals = {
        "observation-only greedy": [0.0, 0.0, 0.0],
        "random feature, zero predictions": [0.0, 0.0, 0.0],
        "oracle": [0.0, 0.0, 0.0],
        "correct-share hedger": [0.0, 0.0, 0.0],
    }
    strategies = (
        ("observation-only greedy", greedy_submission),
        ("random feature, zero predictions", random_submission),
        ("oracle", oracle_submission),
        ("correct-share hedger", hedger_submission),
    )
    for seed in range(seed_count):
        episode = Episode(seed)
        for name, make_submission in strategies:
            scores = score_submission(episode, make_submission(episode, seed))
            for index, value in enumerate(scores):
                totals[name][index] += value
    return {
        name: tuple(value / seed_count for value in sums)
        for name, sums in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    for name, (score, shares, prediction) in evaluate(args.seeds).items():
        print(
            f"{name}: mean score={score:.3f}, "
            f"share recovery={shares:.3f}, "
            f"prediction={prediction:.3f} (n={args.seeds})"
        )


if __name__ == "__main__":
    main()
