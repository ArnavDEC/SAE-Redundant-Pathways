"""Grade causal-share estimates and held-out intervention predictions."""

import json
import math
from pathlib import Path

from private_model import Episode, MAX_REPORTED_FEATURES

EXPECTED_TESTS = ("ablate", "steer", "patch")
MAX_INTERVAL_BOUND = 1e6
MAX_SUBMISSION_BYTES = 65536
SHARE_WEIGHT = 0.5
PREDICTION_WEIGHT = 0.5


def _interval_score(lower: float, upper: float, target: float) -> float:
    """Compute the Winkler interval score for an 80% prediction interval.

    The score is the interval width plus a penalty proportional to how far the
    target falls outside the interval. Lower is better.

    Args:
        lower: Lower bound of the predicted interval.
        upper: Upper bound of the predicted interval.
        target: True held- out effect.

    Returns:
        The interval score, in the same absolute units as the effect.
    """
    alpha = 0.2
    score = upper - lower
    if target < lower:
        score += (2 / alpha) * (lower - target)
    elif target > upper:
        score += (2 / alpha) * (target - upper)
    return score


def score_share_vector(predicted: list[float], target: list[float]) -> float:
    """Score a predicted signed share vector against the hidden target.

    Combines the L1 distance between the two vectors with a penalty for total
    claimed mass differing from 1.

    Args:
        predicted: Submitted shares, one entry per SAE unit.
        target: Hidden true shares, which must have unit L1 mass.

    Returns:
        Share recovery in [0, 1], where 1 is a perfect match.

    Raises:
        ValueError: If the vectors differ in length or the target is not
            unit L1 mass.
    """
    if len(predicted) != len(target):
        raise ValueError("predicted and target share vectors must have equal length")
    target_mass = math.fsum(abs(value) for value in target)
    if not math.isclose(target_mass, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("target signed shares must have unit L1 mass")
    normalized_target = [value / target_mass for value in target]
    prediction_error = math.fsum(
        abs(actual - estimate)
        for actual, estimate in zip(normalized_target, predicted, strict=True)
    )
    mass_error = abs(math.fsum(abs(value) for value in predicted) - 1.0)
    score = max(0.0, 1.0 - 0.5 * (prediction_error + mass_error))
    return 0.0 if score < 1e-12 else score


def grade_submission(submission: object, episode: Episode) -> dict[str, object]:
    """Validate and score a parsed submission against a hidden episode.

    The final score is SHARE_WEIGHT * share recovery plus PREDICTION_WEIGHT *
    (share recovery * held-out prediction), so prediction credit is gated on
    identifying the right units.

    Args:
        submission: Parsed JSON with "components" (feature shares) and
            "predictions" (ablate, steer, patch intervals).
        episode: The hidden episode providing ground-truth shares and effects.

    Returns:
        A dict with "score" and "feedback". Malformed submissions return a
        score of 0.0 with an explanatory message.
    """
    if not isinstance(submission, dict):
        return {"score": 0.0, "feedback": "Submission must be a JSON object"}

    components = submission.get("components")
    if (
        not isinstance(components, list)
        or not components
        or len(components) > MAX_REPORTED_FEATURES
    ):
        return {
            "score": 0.0,
            "feedback": f"components must contain 1 to {MAX_REPORTED_FEATURES} feature-share objects",
        }

    predicted_shares = [0.0] * episode.feature_count
    selected_ids: list[int] = []
    for component in components:
        if not isinstance(component, dict):
            return {"score": 0.0, "feedback": "Each component must be an object"}
        feature_id = component.get("feature_id")
        share = component.get("share")
        if (
            not isinstance(feature_id, int)
            or isinstance(feature_id, bool)
            or not isinstance(share, (int, float))
            or isinstance(share, bool)
            or abs(share) > 1
            or not math.isfinite(share)
        ):
            return {
                "score": 0.0,
                "feedback": "Each component needs an integer feature_id and finite share in [-1, 1]",
            }
        try:
            episode._check_feature(feature_id)
        except ValueError as exc:
            return {"score": 0.0, "feedback": str(exc)}
        if feature_id in selected_ids:
            return {"score": 0.0, "feedback": "Duplicate feature IDs are not allowed"}
        selected_ids.append(feature_id)
        predicted_shares[feature_id] = float(share)

    predictions = submission.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(EXPECTED_TESTS):
        return {
            "score": 0.0,
            "feedback": "predictions must contain ablate, steer, and patch intervals",
        }

    interval_score_total = 0.0
    for prediction, expected_test in zip(predictions, EXPECTED_TESTS, strict=True):
        if not isinstance(prediction, dict) or prediction.get("test") != expected_test:
            return {"score": 0.0, "feedback": f"Expected prediction for {expected_test}"}
        lower = prediction.get("lower")
        upper = prediction.get("upper")
        if (
            not isinstance(lower, (int, float))
            or isinstance(lower, bool)
            or not isinstance(upper, (int, float))
            or isinstance(upper, bool)
            or lower > upper
        ):
            return {
                "score": 0.0,
                "feedback": f"{expected_test} interval must contain finite lower <= upper",
            }
        try:
            lower_float, upper_float = float(lower), float(upper)
        except OverflowError:
            return {
                "score": 0.0,
                "feedback": f"{expected_test} interval bounds must be finite",
            }
        if not math.isfinite(lower_float) or not math.isfinite(upper_float):
            return {
                "score": 0.0,
                "feedback": f"{expected_test} interval bounds must be finite",
            }
        if (
            abs(lower_float) > MAX_INTERVAL_BOUND
            or abs(upper_float) > MAX_INTERVAL_BOUND
        ):
            return {
                "score": 0.0,
                "feedback": (
                    f"{expected_test} interval bounds must be within "
                    f"[-{MAX_INTERVAL_BOUND:g}, {MAX_INTERVAL_BOUND:g}]"
                ),
            }
        target = episode.heldout_effect(expected_test, selected_ids)
        interval_score_total += _interval_score(lower_float, upper_float, target)

    try:
        share_score = score_share_vector(predicted_shares, episode.heldout_shares())
    except ValueError as exc:
        return {"score": 0.0, "feedback": f"Invalid hidden share target: {exc}"}
    prediction_score = max(0.0, 1.0 - interval_score_total / len(EXPECTED_TESTS))
    prediction_credit = share_score * prediction_score
    score = SHARE_WEIGHT * share_score + PREDICTION_WEIGHT * prediction_credit
    return {
        "score": score,
        "feedback": {
            "causal_share_recovery": share_score,
            "heldout_prediction": prediction_score,
            "weighted_prediction_credit": prediction_credit,
            "mean_interval_score": interval_score_total / len(EXPECTED_TESTS),
            "note": (
                "The score weights causal-share recovery and share-weighted "
                "prediction credit equally. Rationale is not scored."
            ),
        },
    }


def grade() -> dict[str, object]:
    """Load the agent's submission and the hidden episode seed, then grade.

    Reads /workdir/submission.json and /grader/episode_seed.json.

    Returns:
        The result of grade_submission with "episode_level" attached, or a
        score of 0.0 and an explanation if the inputs are missing, oversized,
        or unreadable.
    """
    submission_path = Path("/workdir/submission.json")
    if not submission_path.is_file():
        return {"score": 0.0, "feedback": "Missing /workdir/submission.json"}

    try:
        if submission_path.stat().st_size > MAX_SUBMISSION_BYTES:
            return {
                "score": 0.0,
                "feedback": (
                    f"Submission exceeds the {MAX_SUBMISSION_BYTES}-byte limit"
                ),
            }
        submission = json.loads(submission_path.read_text())
        episode_seed = json.loads(Path("/grader/episode_seed.json").read_text())["seed"]
        episode = Episode(episode_seed)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        return {"score": 0.0, "feedback": f"Could not read valid grader input: {exc}"}
    result = grade_submission(submission, episode)
    result["episode_level"] = episode.episode_level
    return result