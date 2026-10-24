#!/usr/bin/env python3
"""Run baseline submissions through the task image's production grade() path."""

import argparse
import json
import statistics
from pathlib import Path

import evaluate
import grader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    strategies = (
        ("observation-only greedy", evaluate.greedy_submission),
        ("random feature, zero predictions", evaluate.random_submission),
        ("oracle", evaluate.oracle_submission),
        ("correct-share hedger", evaluate.hedger_submission),
    )
    seed_path = Path("/grader/episode_seed.json")
    submission_path = Path("/workdir/submission.json")
    Path("/workdir").mkdir(exist_ok=True)
    totals = {name: [[], [], []] for name, _ in strategies}
    levels: dict[str, int] = {}

    for seed in range(args.seeds):
        episode = evaluate.Episode(seed)
        levels[episode.episode_level] = levels.get(episode.episode_level, 0) + 1
        seed_path.write_text(json.dumps({"seed": seed}))
        for name, make_submission in strategies:
            submission_path.write_text(
                json.dumps(make_submission(episode, seed), allow_nan=False)
            )
            result = grader.grade()
            score = result.get("score")
            feedback = result.get("feedback")
            if not isinstance(score, (int, float)) or not isinstance(feedback, dict):
                raise RuntimeError(f"Production grader returned invalid result: {result}")
            totals[name][0].append(float(score))
            totals[name][1].append(float(feedback["causal_share_recovery"]))
            totals[name][2].append(float(feedback["heldout_prediction"]))

    print(f"Image grader() baseline evaluation (n={args.seeds})")
    print("Episode counts:", json.dumps(levels, sort_keys=True))
    for name, (score_values, share_values, prediction_values) in totals.items():
        print(
            f"{name}: mean score={statistics.fmean(score_values):.3f}, "
            f"share recovery={statistics.fmean(share_values):.3f}, "
            f"prediction={statistics.fmean(prediction_values):.3f} "
            f"(n={args.seeds})"
        )


if __name__ == "__main__":
    main()
