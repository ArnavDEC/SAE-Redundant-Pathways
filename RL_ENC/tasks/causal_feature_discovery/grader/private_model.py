"""Hidden attention model, SAE feature space, and held-out episode evaluation."""

import math
import random
import statistics

import torch

from transformer_sae import (
    MODEL_FEATURES,
    MODEL_WIDTH,
    SAE_WIDTH,
    SAEFit,
    TinyAttentionModel,
    estimate_causal_shares,
    features_to_tokens,
    reference_sae,
    sample_contrast_vectors,
    sample_factorial_feature_vectors,
    sample_feature_vectors,
)
from polysemantic_prototype import make_polysemantic_sae
from redundant_pathway_prototype import (
    ALTERNATE_PATH_UNIT,
    PRIMARY_PATH_UNIT,
    make_redundant_pathway_sae,
)

MAX_CALLS = 22
MAX_INPUTS = 64
MAX_REPORTED_FEATURES = 8
VALID_DISTRIBUTIONS = {"train", "decorrelated", "long"}
MODEL_ROTATION_SEED = 0x51AE
UINT64_MASK = (1 << 64) - 1


def _summary(values: list[float]) -> dict[str, float]:
    """Summarize a list of measurements.

    Args:
        values: Per-sample effect measurements.

    Returns:
        A dict with the mean, the standard error (0.0 for a single value),
        and the sample count n.
    """
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "standard_error": standard_error, "n": len(values)}


class Episode:
    """A single hidden task instance, fully determined by its seed.

    Builds the toy attention model and its SAE, chooses the episode level
    (base, polysemantic, or redundant), swaps around the SAE unit IDs so nothing
    carries over between episodes, and computes the hidden share and effect
    targets used for grading. Agents interact only through observe, factorial,
    ablate, steer, and patch; the grader uses heldout_shares and
    heldout_effect.
    """

    def __init__(self, seed: int) -> None:
        """Construct the episode.

        Args:
            seed: Episode seed. Determines the causal concept, level, gain
                (drawn from 1.5 to 2.5), and unit permutation.

        Raises:
            ValueError: If the causal concept has a near-zero held-out effect.
        """
        rng = random.Random(seed)
        self.seed = seed
        self.causal_concept = rng.randrange(MODEL_FEATURES)
        self.polysemantic_nuisance = (self.causal_concept + 2) % MODEL_FEATURES
        self.redundant_concept = (self.causal_concept + 6) % MODEL_FEATURES
        self.polysemantic_level = bool(
            random.Random(seed ^ 0xC0FFEE5EED).getrandbits(1)
        )
        self.redundant_pathway_level = not self.polysemantic_level and bool(
            random.Random(seed ^ 0xA17E2A7E).getrandbits(1)
        )
        self.episode_level = (
            "polysemantic" if self.polysemantic_level else
            "redundant" if self.redundant_pathway_level else "base"
        )
        self.gain = rng.uniform(1.5, 2.5)
        self.feature_count = SAE_WIDTH
        self.model = TinyAttentionModel(
            MODEL_ROTATION_SEED,
            self.causal_concept,
            self.gain,
            redundant_concept=(
                self.redundant_concept if self.redundant_pathway_level else None
            ),
        )
        self.sae = reference_sae()
        if self.redundant_pathway_level:
            self.sae = make_redundant_pathway_sae(
                self.sae,
                self.model.concept_directions,
                self.causal_concept,
                self.redundant_concept,
            )
        elif self.polysemantic_level:
            self.sae = make_polysemantic_sae(
                self.sae,
                self.model.concept_directions,
                self.causal_concept,
                self.polysemantic_nuisance,
            )
        self.feature_permutation = list(range(self.feature_count))
        rng.shuffle(self.feature_permutation)

        if self.redundant_pathway_level:
            shares_by_internal_id = [0.0] * self.feature_count
            shares_by_internal_id[PRIMARY_PATH_UNIT] = 0.5
            shares_by_internal_id[ALTERNATE_PATH_UNIT] = 0.5
            self.causal_shares = [
                shares_by_internal_id[index] for index in self.feature_permutation
            ]
            model_effect = self.gain
            contributions = [share * model_effect for share in shares_by_internal_id]
        else:
            fit = SAEFit(
                model=self.model,
                sae=self.sae,
                explained_variance=0.0,
                relative_reconstruction_rmse=0.0,
            )
            share_estimate = estimate_causal_shares(
                fit,
                self.causal_concept,
                seed=self._seed(0, 9001),
                samples_per_condition=512,
                distribution="decorrelated",
            )
            self.causal_shares = [
                float(share_estimate.signed_shares[index])
                for index in self.feature_permutation
            ]
            model_effect = share_estimate.model_effect
            contributions = [
                float(share_estimate.signed_contributions[index])
                for index in range(self.feature_count)
            ]
        self._heldout_shares = self.causal_shares.copy()
        if abs(model_effect) < 1e-8:
            raise ValueError("causal concept must have a nonzero held-out effect")
        self._heldout_patch_effects = [
            contributions[index] / model_effect
            for index in self.feature_permutation
        ]
        self.causal_feature = max(
            range(self.feature_count), key=lambda index: abs(self.causal_shares[index])
        )

    def _seed(self, nonce: int, salt: int = 0) -> int:
        """Derive a deterministic 64-bit seed for one sampling call.

        Args:
            nonce: Per-call counter that varies the draw between experiments.
            salt: Constant that separates different uses of the same nonce.

        Returns:
            A 64-bit seed mixed from the episode seed, nonce, and salt.
        """
        return (
            self.seed ^ (nonce * 0x9E3779B97F4A7C15) ^ salt
        ) & UINT64_MASK

    @staticmethod
    def _check_distribution(distribution: str) -> None:
        """Validate a distribution name.

        Args:
            distribution: One of "train", "decorrelated", or "long".

        Raises:
            ValueError: If the distribution is not supported.
        """
        if distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(f"unsupported distribution: {distribution}")

    def _check_feature(self, feature_id: int) -> None:
        """Validate a reported (permuted) feature ID.

        Args:
            feature_id: Unit ID as exposed to the agent.

        Raises:
            ValueError: If the ID is outside [0, feature_count).
        """
        if not 0 <= feature_id < self.feature_count:
            raise ValueError(f"feature_id must be between 0 and {self.feature_count - 1}")

    def _internal_feature(self, feature_id: int) -> int:
        """Map a reported feature ID to the internal SAE unit index.

        Args:
            feature_id: Unit ID as exposed to the agent.

        Returns:
            The internal unit index after undoing the per-episode shuffle.

        Raises:
            ValueError: If the ID is out of range.
        """
        self._check_feature(feature_id)
        return self.feature_permutation[feature_id]

    def _internal_features(self, feature_ids: list[int]) -> list[int]:
        """Validate a set of reported feature IDs and map them to internal indices.

        Args:
            feature_ids: Unit IDs as exposed to the agent.

        Returns:
            Internal unit indices, in the same order.

        Raises:
            ValueError: If the list is empty, longer than
                MAX_REPORTED_FEATURES, contains duplicates, or has an
                out-of-range ID.
        """
        if not feature_ids:
            raise ValueError("at least one feature ID is required")
        if len(feature_ids) > MAX_REPORTED_FEATURES:
            raise ValueError(
                f"at most {MAX_REPORTED_FEATURES} feature IDs can be intervened on"
            )
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("duplicate feature IDs are not allowed")
        return [self._internal_feature(feature_id) for feature_id in feature_ids]

    def _measure(
        self, distribution: str, count: int, nonce: int, salt: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample inputs and run them through the model and SAE.

        In the redundant level, the second concept is resampled independently
        at 50% so both routes are exercised.

        Args:
            distribution: "train", "decorrelated", or "long".
            count: Number of inputs to sample.
            nonce: Per-call counter for seeding.
            salt: Extra seed separation between different callers.

        Returns:
            A tuple of (concept vectors, SAE codes, output logits), each with
            one row per input.

        Raises:
            ValueError: If the distribution is not supported.
        """
        self._check_distribution(distribution)
        vectors = sample_feature_vectors(
            self._seed(nonce, salt), count, self.causal_concept, distribution
        )
        if self.redundant_pathway_level:
            generator = torch.Generator(device="cpu").manual_seed(
                self._seed(nonce, salt ^ 0xBEEF)
            )
            vectors[:, self.redundant_concept] = (
                torch.rand(count, generator=generator) < 0.5
            ).float()
        repetitions = 2 if distribution == "long" else 1
        tokens = features_to_tokens(vectors, repetitions=repetitions)
        with torch.no_grad():
            activations = self.model.activations(tokens)
            codes = self.sae.encode(activations)
            logits = self.model.logits_from_activations(activations)
        return vectors, codes, logits

    def _redundant_ablation_effects(
        self, vectors: torch.Tensor, codes: torch.Tensor, internal_ids: list[int]
    ) -> torch.Tensor:
        """Measure ablation effects by actually rerunning the readout.

        Needed in the redundant level because the readout saturates (max), so
        the effect is not linear in the decoder directions.

        Args:
            vectors: Concept vectors for the sampled inputs.
            codes: SAE codes for the same inputs.
            internal_ids: Internal indices of the units to ablate.

        Returns:
            Per-input drop in logit from removing the units' contribution.
        """
        activations = self.model.activations(features_to_tokens(vectors))
        removed = codes[:, internal_ids] @ self.sae.decoder[:, internal_ids].T
        clean_logits = self.model.logits_from_activations(activations)
        ablated_logits = self.model.logits_from_activations(activations - removed)
        return clean_logits - ablated_logits

    def _redundant_patch_values(
        self, internal_ids: list[int], count: int, nonce: int, salt: int
    ) -> torch.Tensor:
        """Measure patch recovery for the redundant level.

        Builds backgrounds with neither route on, and sources with exactly one
        route on (chosen at random per input). Patches the units' codes from
        source into background and reports the fraction of the source-vs-
        background logit gap recovered.

        Args:
            internal_ids: Internal indices of the units to patch.
            count: Number of input pairs.
            nonce: Per-call counter for seeding.
            salt: Extra seed separation between callers.

        Returns:
            Per-pair recovery fraction (patched gap / full gap).
        """
        background = sample_feature_vectors(
            self._seed(nonce, salt),
            count,
            self.causal_concept,
            "decorrelated",
        )
        background[:, self.causal_concept] = 0.0
        background[:, self.redundant_concept] = 0.0
        generator = torch.Generator(device="cpu").manual_seed(
            self._seed(nonce, salt ^ 0xFACE)
        )
        primary_source = torch.rand(count, generator=generator) < 0.5
        source = background.clone()
        source[primary_source, self.causal_concept] = 1.0
        source[~primary_source, self.redundant_concept] = 1.0

        with torch.no_grad():
            background_activations = self.model.activations(
                features_to_tokens(background)
            )
            source_activations = self.model.activations(features_to_tokens(source))
            background_codes = self.sae.encode(background_activations)
            source_codes = self.sae.encode(source_activations)
            patched_activations = background_activations + (
                source_codes[:, internal_ids] - background_codes[:, internal_ids]
            ) @ self.sae.decoder[:, internal_ids].T
            background_logits = self.model.logits_from_activations(
                background_activations
            )
            source_logits = self.model.logits_from_activations(source_activations)
            patched_logits = self.model.logits_from_activations(patched_activations)
        denominator = (source_logits - background_logits).clamp_min(1e-8)
        return (patched_logits - background_logits) / denominator

    def observe(self, distribution: str, count: int, nonce: int) -> dict[str, object]:
        """Return observational data: SAE unit activations and noisy logit diffs.

        Args:
            distribution: "train", "decorrelated", or "long".
            count: Number of inputs to sample.
            nonce: Per-call counter for seeding.

        Returns:
            A dict with the distribution name and a list of observations,
            each holding the unit activations (in reported ID order) and a
            logit difference with Gaussian noise (sd 0.2).
        """
        vectors, codes, logits = self._measure(distribution, count, nonce)
        rng = random.Random(self._seed(nonce, 100))
        observations = []
        for row in range(count):
            observed_features = [
                float(codes[row, internal_id])
                for internal_id in self.feature_permutation
            ]
            observations.append(
                {
                    "features": observed_features,
                    "logit_diff": float(logits[row]) + rng.gauss(0.0, 0.2),
                }
            )
        return {"distribution": distribution, "observations": observations}

    def factorial(
        self,
        count: int,
        nonce: int,
        feature_ids: list[int] | None = None,
    ) -> dict[str, object]:
        """Return matched target/second-concept condition observations.

        Samples inputs for four conditions (neither, target_only, second_only,
        both) on paired backgrounds. The second concept is the redundant
        concept in the redundant level, and the nuisance concept otherwise.
        The condition labels only describe which input is present, not
        whether it is causal.

        Args:
            count: Inputs per condition.
            nonce: Per-call counter for seeding.
            feature_ids: Optional units whose ablation effect should also be
                reported for each condition.

        Returns:
            A dict with the experiment name, paired_backgrounds flag, the
            feature IDs, and per-condition observations (plus ablation
            summaries when feature_ids is given).

        Raises:
            ValueError: If count is outside 1 to MAX_INPUTS // 4, or if the
                feature IDs are invalid.
        """
        if count < 1 or count * 4 > MAX_INPUTS:
            raise ValueError(
                f"count must be from 1 to {MAX_INPUTS // 4} per condition"
            )
        internal_ids = (
            self._internal_features(feature_ids) if feature_ids is not None else None
        )
        vectors_by_condition = sample_factorial_feature_vectors(
            self._seed(nonce, 201),
            count,
            self.causal_concept,
            (
                self.redundant_concept
                if self.redundant_pathway_level
                else self.polysemantic_nuisance
            ),
        )
        rng = random.Random(self._seed(nonce, 202))
        conditions: dict[str, object] = {}
        for label, vectors in vectors_by_condition.items():
            tokens = features_to_tokens(vectors)
            with torch.no_grad():
                activations = self.model.activations(tokens)
                codes = self.sae.encode(activations)
                logits = self.model.logits_from_activations(activations)
            observations = []
            for row in range(count):
                exposed_features = [
                    float(codes[row, internal_id])
                    for internal_id in self.feature_permutation
                ]
                observations.append({
                    "features": exposed_features,
                    "logit_diff": float(logits[row]) + rng.gauss(0.0, 0.2),
                })
            condition_data: dict[str, object] = {"observations": observations}
            if internal_ids is not None:
                if self.redundant_pathway_level:
                    effects = self._redundant_ablation_effects(
                        vectors, codes, internal_ids
                    )
                    ablation_effects = [
                        float(effects[row]) + rng.gauss(0.0, 0.1)
                        for row in range(count)
                    ]
                else:
                    decoder_effects = self.model.readout @ self.sae.decoder[:, internal_ids]
                    ablation_effects = [
                        float(codes[row, internal_ids] @ decoder_effects)
                        + rng.gauss(0.0, 0.1)
                        for row in range(count)
                    ]
                condition_data["ablation"] = {
                    "feature_ids": feature_ids,
                    **_summary(ablation_effects),
                }
            conditions[label] = condition_data
        return {
            "experiment": "factorial",
            "paired_backgrounds": True,
            "feature_ids": feature_ids,
            "conditions": conditions,
        }

    def ablate(
        self, feature_ids: list[int], distribution: str, count: int, nonce: int
    ) -> dict[str, object]:
        """Ablate a set of units jointly and report the logit effect.

        Args:
            feature_ids: Reported unit IDs to ablate (up to
                MAX_REPORTED_FEATURES, no duplicates).
            distribution: "train", "decorrelated", or "long".
            count: Number of inputs to sample.
            nonce: Per-call counter for seeding.

        Returns:
            A dict with the intervention name, feature IDs, and the mean,
            standard error, and n of the per-input effects (noise sd 0.1).

        Raises:
            ValueError: If the feature IDs or distribution are invalid.
        """
        internal_ids = self._internal_features(feature_ids)
        vectors, codes, logits = self._measure(distribution, count, nonce, 1)
        rng = random.Random(self._seed(nonce, 101))
        if self.redundant_pathway_level:
            measured = self._redundant_ablation_effects(vectors, codes, internal_ids)
            effects = [float(value) + rng.gauss(0.0, 0.1) for value in measured]
        else:
            decoder_effects = self.model.readout @ self.sae.decoder[:, internal_ids]
            effects = [
                float(codes[row, internal_ids] @ decoder_effects)
                + rng.gauss(0.0, 0.1)
                for row in range(count)
            ]
        return {"intervention": "ablate", "feature_ids": feature_ids, **_summary(effects)}

    def steer(self, feature_ids: list[int], scale: float, count: int, nonce: int) -> dict[str, object]:
        """Add a scaled copy of the units' decoder directions and report the effect.

        Args:
            feature_ids: Reported unit IDs to steer (up to
                MAX_REPORTED_FEATURES, no duplicates).
            scale: Steering coefficient, finite and within [-5, 5].
            count: Number of samples.
            nonce: Per-call counter for seeding.

        Returns:
            A dict with the intervention name, feature IDs, scale, and the
            mean, standard error, and n of the effects (noise sd 0.1).

        Raises:
            ValueError: If the feature IDs are invalid or scale is not finite
                or exceeds 5 in magnitude.
        """
        internal_ids = self._internal_features(feature_ids)
        if not math.isfinite(scale) or abs(scale) > 5:
            raise ValueError("scale must be finite and between -5 and 5")
        rng = random.Random(self._seed(nonce, 102))
        if self.redundant_pathway_level:
            vectors, _, clean_logits = self._measure(
                "decorrelated", count, nonce, 3
            )
            activations = self.model.activations(features_to_tokens(vectors))
            steered_activations = activations + scale * self.sae.decoder[
                :, internal_ids
            ].sum(dim=1)
            steered_logits = self.model.logits_from_activations(steered_activations)
            effects = [
                float(steered_logits[row] - clean_logits[row]) + rng.gauss(0.0, 0.1)
                for row in range(count)
            ]
        else:
            decoder_effect = float(
                (self.model.readout @ self.sae.decoder[:, internal_ids]).sum()
            )
            effects = [
                scale * decoder_effect + rng.gauss(0.0, 0.1)
                for _ in range(count)
            ]
        return {
            "intervention": "steer",
            "feature_ids": feature_ids,
            "scale": scale,
            **_summary(effects),
        }

    def patch(self, feature_ids: list[int], count: int, nonce: int) -> dict[str, object]:
        """Patch units from a target-on input into a target-off input.

        Reports the fraction of the full output gap recovered by patching
        only the chosen units' codes.

        Args:
            feature_ids: Reported unit IDs to patch (up to
                MAX_REPORTED_FEATURES, no duplicates).
            count: Number of input pairs.
            nonce: Per-call counter for seeding.

        Returns:
            A dict with the intervention name, feature IDs, and the mean,
            standard error, and n of the recovery fractions (noise sd 0.05).

        Raises:
            ValueError: If the feature IDs are invalid.
        """
        internal_ids = self._internal_features(feature_ids)
        if self.redundant_pathway_level:
            values = self._redundant_patch_values(internal_ids, count, nonce, 103)
            rng = random.Random(self._seed(nonce, 105))
            noisy_values = [
                float(value) + rng.gauss(0.0, 0.05) for value in values
            ]
            return {
                "intervention": "patch",
                "feature_ids": feature_ids,
                **_summary(noisy_values),
            }
        active = sample_contrast_vectors(
            self._seed(nonce, 103),
            count,
            self.causal_concept,
            True,
            distribution="decorrelated",
        )
        inactive = sample_contrast_vectors(
            self._seed(nonce, 104),
            count,
            self.causal_concept,
            False,
            distribution="decorrelated",
        )
        with torch.no_grad():
            active_codes = self.sae.encode(
                self.model.activations(features_to_tokens(active))
            )
            inactive_codes = self.sae.encode(
                self.model.activations(features_to_tokens(inactive))
            )
        decoder_effects = self.model.readout @ self.sae.decoder[:, internal_ids]
        recovery_mean = (
            float(
                (
                    active_codes[:, internal_ids].mean(dim=0)
                    - inactive_codes[:, internal_ids].mean(dim=0)
                )
                @ decoder_effects
            )
            / self.gain
        )
        rng = random.Random(self._seed(nonce, 105))
        values = [recovery_mean + rng.gauss(0.0, 0.05) for _ in range(count)]
        return {"intervention": "patch", "feature_ids": feature_ids, **_summary(values)}

    def heldout_shares(self) -> list[float]:
        """Return the hidden signed share target, in reported ID order.

        Returns:
            A copy of the share vector, with unit L1 mass.
        """
        return self._heldout_shares.copy()

    def heldout_effect(self, test: str, feature_ids: list[int]) -> float:
        """Compute the hidden ground-truth effect of a joint intervention.
        Evaluated on the decorrelated distribution (2048 samples for the
        sampled cases). Ablation and steering are noise-free means; steering
        uses a fixed scale of 0.5; patch is the fraction of the output gap
        recovered.

        Args:
            test: One of "ablate", "steer", or "patch".
            feature_ids: Reported unit IDs in the intervened set.

        Returns:
            The true held-out effect for that test and unit set.

        Raises:
            ValueError: If the test name is unsupported, the list is empty,
                or the feature IDs are invalid.
        """
        if test not in {"ablate", "steer", "patch"}:
            raise ValueError(f"unsupported held-out test: {test}")
        if not feature_ids:
            raise ValueError("at least one feature ID is required")
        internal_ids = self._internal_features(feature_ids)
        if self.redundant_pathway_level:
            if test == "ablate":
                vectors, codes, _ = self._measure(
                    "decorrelated", 2048, nonce=2, salt=14001
                )
                return float(
                    self._redundant_ablation_effects(
                        vectors, codes, internal_ids
                    ).mean()
                )
            if test == "steer":
                vectors, _, clean_logits = self._measure(
                    "decorrelated", 2048, nonce=2, salt=14002
                )
                activations = self.model.activations(features_to_tokens(vectors))
                steered = activations + 0.5 * self.sae.decoder[:, internal_ids].sum(dim=1)
                steered_logits = self.model.logits_from_activations(steered)
                return float((steered_logits - clean_logits).mean())
            return float(
                self._redundant_patch_values(internal_ids, 2048, nonce=2, salt=14003).mean()
            )
        decoder_effects = self.model.readout @ self.sae.decoder[:, internal_ids]
        if test == "steer":
            return float(decoder_effects.sum()) * 0.5
        if test == "patch":
            return sum(self._heldout_patch_effects[feature_id] for feature_id in feature_ids)

        _, codes, _ = self._measure(
            "decorrelated", 2048, nonce=2, salt=14001
        )
        return float((codes[:, internal_ids] @ decoder_effects).mean())