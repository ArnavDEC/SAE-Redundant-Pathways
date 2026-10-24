import unittest

import torch

from transformer_sae import (
    SAE_TOP_K,
    SAE_WIDTH,
    TinyAttentionModel,
    estimate_causal_shares,
    features_to_tokens,
    fit_sae,
    sample_feature_vectors,
)


class TransformerSAETests(unittest.TestCase):
    def test_attention_model_reads_the_planted_concept(self):
        causal_concept = 4
        gain = 2.25
        model = TinyAttentionModel(seed=123, causal_concept=causal_concept, gain=gain)
        vectors = sample_feature_vectors(seed=456, count=64, causal_concept=causal_concept)
        activations, logit_diff = model(features_to_tokens(vectors))

        torch.testing.assert_close(
            logit_diff, gain * vectors[:, causal_concept], atol=1e-5, rtol=1e-5
        )
        self.assertEqual(tuple(activations.shape), (64, 24))
        torch.testing.assert_close(
            activations, model.activations(features_to_tokens(vectors, repetitions=2))
        )

    def test_train_has_a_sparse_correlated_decoy_removed_by_distribution_shift(self):
        causal_concept = 4
        decoy_concept = (causal_concept + 1) % 12
        train = sample_feature_vectors(4561, 10000, causal_concept, "train")
        decorrelated = sample_feature_vectors(
            4561, 10000, causal_concept, "decorrelated"
        )

        train_correlation = torch.corrcoef(
            train[:, [causal_concept, decoy_concept]].T
        )[0, 1]
        decorrelated_correlation = torch.corrcoef(
            decorrelated[:, [causal_concept, decoy_concept]].T
        )[0, 1]

        self.assertGreater(float(train_correlation), 0.45)
        self.assertLess(abs(float(decorrelated_correlation)), 0.04)
        self.assertLess(float(train[:, decoy_concept].mean()), 0.45)
        self.assertAlmostEqual(float(train[:, causal_concept].mean()), 0.5, delta=0.03)
        self.assertAlmostEqual(
            float(decorrelated[:, causal_concept].mean()), 0.5, delta=0.03
        )

        # Three individually strong proxies are planted, but all remain
        # noncausal. Check the other two as well as the historical +1 slot.
        for offset in (1, 3, 4):
            proxy = (causal_concept + offset) % 12
            train_corr = torch.corrcoef(train[:, [causal_concept, proxy]].T)[0, 1]
            shifted_corr = torch.corrcoef(
                decorrelated[:, [causal_concept, proxy]].T
            )[0, 1]
            self.assertGreater(float(train_corr), 0.75)
            self.assertLess(abs(float(shifted_corr)), 0.04)

        model = TinyAttentionModel(seed=987, causal_concept=causal_concept, gain=2.0)
        changed_decoy = train.clone()
        for offset in (1, 3, 4):
            proxy = (causal_concept + offset) % 12
            changed_decoy[:, proxy] = 1.0 - changed_decoy[:, proxy]
        original_logits = model.logits_from_activations(
            model.activations(features_to_tokens(train))
        )
        changed_logits = model.logits_from_activations(
            model.activations(features_to_tokens(changed_decoy))
        )
        torch.testing.assert_close(original_logits, changed_logits)

    def test_sae_trains_and_returns_topk_sparse_codes(self):
        fit = fit_sae(
            seed=777,
            causal_concept=2,
            gain=2.0,
            training_examples=256,
            steps=10,
        )
        vectors = sample_feature_vectors(seed=888, count=32, causal_concept=2)
        activations = fit.model.activations(features_to_tokens(vectors))
        codes = fit.sae.encode(activations)

        self.assertEqual(tuple(codes.shape), (32, SAE_WIDTH))
        self.assertTrue(torch.all((codes > 0).sum(dim=1) <= SAE_TOP_K))
        self.assertTrue(torch.isfinite(codes).all())
        self.assertGreater(fit.explained_variance, 0.0)
        self.assertLess(fit.relative_reconstruction_rmse, 1.0)

    def test_causal_effect_can_be_attributed_as_a_signed_share_vector(self):
        fit = fit_sae(
            seed=992,
            causal_concept=3,
            gain=2.0,
            training_examples=256,
            steps=10,
        )
        estimate = estimate_causal_shares(
            fit, causal_concept=3, seed=993, samples_per_condition=128
        )

        self.assertAlmostEqual(
            float(estimate.signed_shares.abs().sum()), 1.0, places=5
        )
        self.assertAlmostEqual(estimate.model_effect, 2.0, delta=1e-5)
        self.assertGreater(
            int((estimate.signed_contributions.abs() > 1e-6).sum()), 1
        )


if __name__ == "__main__":
    unittest.main()
