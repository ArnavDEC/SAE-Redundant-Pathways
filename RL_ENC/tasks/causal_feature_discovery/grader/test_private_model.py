import random
import unittest

from grader import grade_submission, score_share_vector
from private_model import Episode, MAX_REPORTED_FEATURES


class EpisodeTests(unittest.TestCase):
    def test_polysemantic_episode_exposes_matched_target_nuisance_factorial(self):
        seed = next(
            candidate
            for candidate in range(100)
            if random.Random(candidate ^ 0xC0FFEE5EED).getrandbits(1)
        )
        episode = Episode(seed)
        self.assertTrue(episode.polysemantic_level)
        self.assertEqual(
            episode.polysemantic_nuisance, (episode.causal_concept + 2) % 12
        )

        shared_public_id = episode.feature_permutation.index(47)
        result = episode.factorial(8, 1, [shared_public_id])
        conditions = result["conditions"]
        self.assertEqual(
            set(conditions), {"neither", "target_only", "second_only", "both"}
        )
        activations = {
            label: sum(
                row["features"][shared_public_id] for row in data["observations"]
            )
            / 8
            for label, data in conditions.items()
        }
        self.assertAlmostEqual(activations["neither"], 0.0, delta=0.1)
        self.assertAlmostEqual(activations["target_only"], 1.5, delta=0.1)
        self.assertAlmostEqual(activations["second_only"], 1.5, delta=0.1)
        self.assertAlmostEqual(activations["both"], 3.0, delta=0.1)

        ablations = {
            label: data["ablation"]["mean"] for label, data in conditions.items()
        }
        target_gap_without_nuisance = ablations["target_only"] - ablations["neither"]
        target_gap_with_nuisance = ablations["both"] - ablations["second_only"]
        self.assertAlmostEqual(
            target_gap_without_nuisance, target_gap_with_nuisance, delta=0.15
        )

    def test_episode_is_repeatable(self):
        first = Episode(918273)
        second = Episode(918273)
        self.assertEqual(first.causal_concept, second.causal_concept)
        self.assertEqual(first.gain, second.gain)
        self.assertEqual(first.feature_permutation, second.feature_permutation)
        self.assertEqual(first.causal_shares, second.causal_shares)
        self.assertEqual(first.observe("train", 8, 1), second.observe("train", 8, 1))

    def test_learned_feature_observations_and_interventions_use_signed_effects(self):
        episode = Episode(918273)
        feature_id = episode.causal_feature
        observation = episode.observe("decorrelated", 64, 2)
        self.assertEqual(len(observation["observations"]), 64)
        self.assertEqual(
            len(observation["observations"][0]["features"]), episode.feature_count
        )

        ablation = episode.ablate([feature_id], "decorrelated", 64, 3)
        steering = episode.steer([feature_id], 0.5, 64, 4)
        patch = episode.patch([feature_id], 64, 5)
        self.assertTrue(-2.5 < ablation["mean"] < 2.5)
        self.assertTrue(-2.5 < steering["mean"] < 2.5)
        self.assertTrue(-1.5 < patch["mean"] < 1.5)
        self.assertGreater(ablation["standard_error"], 0)
        self.assertGreater(steering["standard_error"], 0)

    def test_joint_interventions_accept_sets_and_enforce_limits(self):
        episode = Episode(918273)
        feature_ids = [episode.causal_feature, (episode.causal_feature + 1) % episode.feature_count]

        ablation = episode.ablate(feature_ids, "decorrelated", 32, 10)
        steering = episode.steer(feature_ids, 0.5, 32, 11)
        patch = episode.patch(feature_ids, 32, 12)
        self.assertEqual(ablation["feature_ids"], feature_ids)
        self.assertEqual(steering["feature_ids"], feature_ids)
        self.assertEqual(patch["feature_ids"], feature_ids)
        self.assertTrue(all(result["n"] == 32 for result in (ablation, steering, patch)))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            episode.ablate([feature_ids[0], feature_ids[0]], "train", 8, 13)
        with self.assertRaisesRegex(ValueError, "at most"):
            episode.steer(list(range(MAX_REPORTED_FEATURES + 1)), 0.5, 8, 14)

    def test_joint_patch_probe_matches_heldout_fraction_of_model_gap(self):
        episode = Episode(918273)
        feature_ids = sorted(
            range(episode.feature_count),
            key=lambda feature_id: abs(episode._heldout_patch_effects[feature_id]),
            reverse=True,
        )[:MAX_REPORTED_FEATURES]
        measured = episode.patch(feature_ids, 2048, 15)["mean"]
        target = episode.heldout_effect("patch", feature_ids)
        self.assertAlmostEqual(measured, target, delta=0.04)

    def test_share_score_rewards_recovered_signed_mass(self):
        target = [0.7, -0.2, 0.1]
        self.assertEqual(score_share_vector(target, target), 1.0)
        self.assertEqual(score_share_vector([0.0, 0.0, 0.0], target), 0.0)
        self.assertLess(score_share_vector([0.7, 0.2, 0.1], target), 1.0)

    def test_submission_uses_a_limited_set_and_joint_predictions(self):
        episode = Episode(918273)
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
        predictions = [
            {
                "test": test,
                "lower": episode.heldout_effect(test, selected) - 0.01,
                "upper": episode.heldout_effect(test, selected) + 0.01,
            }
            for test in ("ablate", "steer", "patch")
        ]

        result = grade_submission(
            {"components": components, "predictions": predictions}, episode
        )
        feedback = result["feedback"]
        self.assertIsInstance(feedback, dict)
        self.assertGreater(feedback["causal_share_recovery"], 0.5)
        self.assertGreater(feedback["heldout_prediction"], 0.95)

    def test_missing_share_mass_does_not_receive_free_score(self):
        target = [0.6, 0.3, -0.1]
        self.assertEqual(score_share_vector([0.0, 0.0, 0.0], target), 0.0)

    def test_accurate_intervals_without_share_recovery_get_no_credit(self):
        episode = Episode(918273)
        feature_ids = [episode.causal_feature]
        predictions = [
            {
                "test": test,
                "lower": episode.heldout_effect(test, feature_ids),
                "upper": episode.heldout_effect(test, feature_ids),
            }
            for test in ("ablate", "steer", "patch")
        ]
        result = grade_submission(
            {
                "components": [{"feature_id": feature_ids[0], "share": 0.0}],
                "predictions": predictions,
            },
            episode,
        )
        feedback = result["feedback"]
        self.assertIsInstance(feedback, dict)
        self.assertEqual(feedback["heldout_prediction"], 1.0)
        self.assertEqual(feedback["causal_share_recovery"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_wide_intervals_do_not_earn_prediction_credit(self):
        episode = Episode(918273)
        shares = episode.heldout_shares()
        selected = sorted(
            range(episode.feature_count),
            key=lambda feature_id: abs(shares[feature_id]),
            reverse=True,
        )[:MAX_REPORTED_FEATURES]
        result = grade_submission(
            {
                "components": [
                    {"feature_id": feature_id, "share": shares[feature_id]}
                    for feature_id in selected
                ],
                "predictions": [
                    {"test": test, "lower": -1e6, "upper": 1e6}
                    for test in ("ablate", "steer", "patch")
                ],
            },
            episode,
        )

        feedback = result["feedback"]
        self.assertIsInstance(feedback, dict)
        self.assertEqual(feedback["heldout_prediction"], 0.0)
        self.assertAlmostEqual(
            result["score"], 0.5 * feedback["causal_share_recovery"]
        )

    def test_adversarial_interval_values_are_rejected_without_crashing(self):
        episode = Episode(918273)
        feature_id = episode.causal_feature
        for lower, upper in (
            (float("nan"), 1.0),
            (0.0, float("inf")),
            (10**10000, 10**10000),
            (-1e100, 1e100),
        ):
            with self.subTest(lower=type(lower).__name__, upper=type(upper).__name__):
                result = grade_submission(
                    {
                        "components": [{"feature_id": feature_id, "share": 1.0}],
                        "predictions": [
                            {"test": test, "lower": lower, "upper": upper}
                            for test in ("ablate", "steer", "patch")
                        ],
                    },
                    episode,
                )
                self.assertEqual(result["score"], 0.0)
                self.assertIsInstance(result["feedback"], str)

    def test_adversarial_share_components_are_rejected(self):
        episode = Episode(918273)
        valid_predictions = [
            {"test": test, "lower": 0.0, "upper": 0.1}
            for test in ("ablate", "steer", "patch")
        ]
        invalid_components = (
            [{"feature_id": True, "share": 1.0}],
            [{"feature_id": episode.causal_feature, "share": True}],
            [{"feature_id": episode.causal_feature, "share": float("nan")}],
            [
                {"feature_id": episode.causal_feature, "share": 0.5},
                {"feature_id": episode.causal_feature, "share": 0.5},
            ],
            [
                {"feature_id": feature_id, "share": 0.0}
                for feature_id in range(MAX_REPORTED_FEATURES + 1)
            ],
            [{"feature_id": episode.feature_count, "share": 1.0}],
        )
        for components in invalid_components:
            with self.subTest(components=components):
                result = grade_submission(
                    {"components": components, "predictions": valid_predictions},
                    episode,
                )
                self.assertEqual(result["score"], 0.0)
                self.assertIsInstance(result["feedback"], str)


if __name__ == "__main__":
    unittest.main()
