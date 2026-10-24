"""Tiny attention model and sparse autoencoder used for the SAE prototype."""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_FEATURES = 12
MODEL_WIDTH = 24
SAE_WIDTH = 48
SAE_TOP_K = 4
SEQUENCE_LENGTH = MODEL_FEATURES


class TinyAttentionModel(nn.Module):
    """One-head uniform self-attention over concept-token positions.

    Query and key projections are zero, so attention is uniform. The resulting
    residual is the sum of active token embeddings; the unembedding reads one
    planted concept direction to produce the target logit difference.
    """

    def __init__(
        self,
        seed: int,
        causal_concept: int,
        gain: float,
        redundant_concept: int | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= causal_concept < MODEL_FEATURES:
            raise ValueError("causal_concept is outside the model vocabulary")
        if redundant_concept is not None and (
            not 0 <= redundant_concept < MODEL_FEATURES
            or redundant_concept == causal_concept
        ):
            raise ValueError("redundant_concept must be distinct and in the vocabulary")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_matrix = torch.randn(
            MODEL_WIDTH, MODEL_WIDTH, generator=generator, dtype=torch.float64
        )
        rotation, _ = torch.linalg.qr(random_matrix)
        concept_directions = rotation[:, :MODEL_FEATURES].to(torch.float32)
        self.register_buffer("concept_directions", concept_directions)
        token_embedding = torch.cat(
            (torch.zeros(1, MODEL_WIDTH), concept_directions.T), dim=0
        )
        self.register_buffer("token_embedding", token_embedding)
        self.register_buffer(
            "readout", gain * concept_directions[:, causal_concept].clone()
        )
        alternate_readout = (
            gain * concept_directions[:, redundant_concept].clone()
            if redundant_concept is not None
            else torch.zeros(MODEL_WIDTH)
        )
        self.register_buffer("alternate_readout", alternate_readout)
        self.causal_concept = causal_concept
        self.redundant_concept = redundant_concept
        self.gain = gain

    def activations(self, token_ids: Tensor) -> Tensor:
        if (
            token_ids.ndim != 2
            or token_ids.shape[1] < SEQUENCE_LENGTH
            or token_ids.shape[1] % SEQUENCE_LENGTH != 0
        ):
            raise ValueError(
                f"token_ids must have shape (batch, N * {SEQUENCE_LENGTH})"
            )
        embedded = F.embedding(token_ids, self.token_embedding)
        query = torch.zeros_like(embedded[:, -1:, :])
        keys = torch.zeros_like(embedded)
        scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(MODEL_WIDTH)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, embedded).squeeze(1)
        return context * SEQUENCE_LENGTH

    def logits_from_activations(self, activations: Tensor) -> Tensor:
        if self.redundant_concept is not None:
            primary = torch.matmul(activations, self.readout) / self.gain
            alternate = torch.matmul(activations, self.alternate_readout) / self.gain
            return self.gain * torch.maximum(
                primary.clamp(0.0, 1.0), alternate.clamp(0.0, 1.0)
            )
        return torch.matmul(activations, self.readout)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        activations = self.activations(token_ids)
        return activations, self.logits_from_activations(activations)


class TopKSparseAutoencoder(nn.Module):
    def __init__(
        self, input_width: int = MODEL_WIDTH, top_k: int = SAE_TOP_K
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= SAE_WIDTH:
            raise ValueError("top_k must be between 1 and SAE_WIDTH")
        self.top_k = top_k
        self.encoder = nn.Linear(input_width, SAE_WIDTH)
        self.decoder = nn.Parameter(torch.empty(input_width, SAE_WIDTH))
        self.register_buffer("mean", torch.zeros(input_width))
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.normal_(self.decoder, std=1 / math.sqrt(input_width))

    def encode(self, activations: Tensor) -> Tensor:
        values = F.relu(self.encoder(activations - self.mean))
        top_values, top_indices = torch.topk(values, self.top_k, dim=-1)
        sparse = torch.zeros_like(values).scatter(-1, top_indices, top_values)
        return sparse

    def decode(self, features: Tensor) -> Tensor:
        return features @ self.decoder.T + self.mean

    def forward(self, activations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encode(activations)
        return self.decode(features), features


@dataclass(frozen=True)
class SAEFit:
    model: TinyAttentionModel
    sae: TopKSparseAutoencoder
    explained_variance: float
    relative_reconstruction_rmse: float


@dataclass(frozen=True)
class CausalShareEstimate:
    signed_contributions: Tensor
    signed_shares: Tensor
    model_effect: float
    sae_reconstructed_effect: float


def sample_feature_vectors(
    seed: int,
    count: int,
    causal_concept: int = 0,
    distribution: str = "train",
) -> Tensor:
    if distribution not in {"train", "decorrelated", "long"}:
        raise ValueError(f"unsupported distribution: {distribution}")
    if not 0 <= causal_concept < MODEL_FEATURES:
        raise ValueError("causal_concept is outside the model vocabulary")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nuisance_probability = 0.25 if distribution == "decorrelated" else 0.1
    probabilities = torch.full((count, MODEL_FEATURES), nuisance_probability)
    probabilities[:, causal_concept] = 0.5
    vectors = (torch.rand(count, MODEL_FEATURES, generator=generator) < probabilities).float()
    if distribution == "train":
        # Several noncausal concepts are strong train-only proxies for the
        # target. Their independent sampling at inference makes observational
        # ranking especially unreliable without changing the model readout.
        target_active = vectors[:, causal_concept] > 0
        for offset in (1, 3, 4):
            decoy_concept = (causal_concept + offset) % MODEL_FEATURES
            decoy_probability = torch.where(
                target_active,
                torch.full((count,), 0.8),
                torch.full((count,), 0.01),
            )
            vectors[:, decoy_concept] = (
                torch.rand(count, generator=generator) < decoy_probability
            ).float()
    return vectors


def sample_sae_training_vectors(seed: int, count: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(count, MODEL_FEATURES, generator=generator) < 0.15).float()


def sample_contrast_vectors(
    seed: int,
    count: int,
    causal_concept: int,
    active: bool,
    distribution: str = "decorrelated",
) -> Tensor:
    vectors = sample_feature_vectors(
        seed, count, causal_concept=causal_concept, distribution=distribution
    )
    vectors[:, causal_concept] = float(active)
    return vectors


def sample_factorial_feature_vectors(
    seed: int,
    count: int,
    target_concept: int,
    second_concept: int,
    *,
    background_probability: float = 0.25,
) -> dict[str, Tensor]:
    """Sample matched four-way target/second-concept contrasts.

    Every condition shares the same random background concepts, so differences
    isolate the two concept toggles instead of background sampling.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if not 0.0 <= background_probability <= 1.0:
        raise ValueError("background_probability must be between 0 and 1")
    if not 0 <= target_concept < MODEL_FEATURES:
        raise ValueError("target_concept is outside the model vocabulary")
    if not 0 <= second_concept < MODEL_FEATURES:
        raise ValueError("second_concept is outside the model vocabulary")
    if target_concept == second_concept:
        raise ValueError("target and second concepts must be distinct")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    background = (
        torch.rand(count, MODEL_FEATURES, generator=generator)
        < background_probability
    ).float()
    background[:, target_concept] = 0.0
    background[:, second_concept] = 0.0

    conditions: dict[str, Tensor] = {}
    for label, target_active, second_active in (
        ("neither", False, False),
        ("target_only", True, False),
        ("second_only", False, True),
        ("both", True, True),
    ):
        vectors = background.clone()
        vectors[:, target_concept] = float(target_active)
        vectors[:, second_concept] = float(second_active)
        conditions[label] = vectors
    return conditions


def features_to_tokens(vectors: Tensor, repetitions: int = 1) -> Tensor:
    if vectors.ndim != 2 or vectors.shape[1] != MODEL_FEATURES:
        raise ValueError(f"vectors must have shape (batch, {MODEL_FEATURES})")
    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    token_ids = torch.arange(1, MODEL_FEATURES + 1).expand(vectors.shape[0], -1)
    tokens = torch.where(vectors > 0, token_ids, torch.zeros_like(token_ids))
    return tokens.repeat(1, repetitions)


def estimate_causal_shares(
    fit: SAEFit,
    causal_concept: int,
    *,
    seed: int,
    samples_per_condition: int = 2048,
    distribution: str = "decorrelated",
) -> CausalShareEstimate:
    """Attribute the trigger-dependent output gap across learned SAE units."""
    with torch.no_grad():
        active_vectors = sample_contrast_vectors(
            seed, samples_per_condition, causal_concept, True, distribution
        )
        inactive_vectors = sample_contrast_vectors(
            seed ^ 0x6A09E667,
            samples_per_condition,
            causal_concept,
            False,
            distribution,
        )
        active_activations = fit.model.activations(features_to_tokens(active_vectors))
        inactive_activations = fit.model.activations(features_to_tokens(inactive_vectors))
        active_codes = fit.sae.encode(active_activations)
        inactive_codes = fit.sae.encode(inactive_activations)
        code_delta = active_codes.mean(dim=0) - inactive_codes.mean(dim=0)
        decoder_effects = fit.model.readout @ fit.sae.decoder
        contributions = code_delta * decoder_effects
        scale = contributions.abs().sum().clamp_min(1e-8)
        shares = contributions / scale
        model_effect = float(
            (
                fit.model.logits_from_activations(active_activations).mean()
                - fit.model.logits_from_activations(inactive_activations).mean()
            ).item()
        )
        reconstructed_effect = float(contributions.sum().item())
    return CausalShareEstimate(
        signed_contributions=contributions.detach(),
        signed_shares=shares.detach(),
        model_effect=model_effect,
        sae_reconstructed_effect=reconstructed_effect,
    )


def fit_sae(
    seed: int,
    causal_concept: int,
    gain: float,
    *,
    training_examples: int = 2048,
    steps: int = 250,
) -> SAEFit:
    """Train a small top-k SAE on internal residual activations."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = TinyAttentionModel(seed, causal_concept, gain)
    train_vectors = sample_sae_training_vectors(
        seed ^ 0x13579BDF, training_examples
    )
    train_activations = model.activations(features_to_tokens(train_vectors))
    sae = TopKSparseAutoencoder()
    sae.mean.copy_(train_activations.mean(dim=0))
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-3)
    batch_size = min(256, training_examples)
    batch_rng = torch.Generator(device="cpu").manual_seed(seed ^ 0x2468ACE0)

    for _ in range(steps):
        indices = torch.randint(
            training_examples, (batch_size,), generator=batch_rng
        )
        batch = train_activations[indices]
        reconstructed, codes = sae(batch)
        reconstruction_loss = F.mse_loss(reconstructed, batch)
        sparsity_loss = codes.abs().mean()
        loss = reconstruction_loss + 1e-4 * sparsity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sae.decoder.div_(
                torch.linalg.vector_norm(sae.decoder, dim=0, keepdim=True).clamp_min(1e-8)
            )

    with torch.no_grad():
        reconstructed, _ = sae(train_activations)
        residual = train_activations - reconstructed
        variance = train_activations.var(unbiased=False).clamp_min(1e-8)
        explained_variance = 1.0 - residual.var(unbiased=False) / variance
        relative_rmse = torch.sqrt(residual.square().mean()) / torch.sqrt(
            train_activations.square().mean()
        ).clamp_min(1e-8)
    return SAEFit(
        model=model,
        sae=sae,
        explained_variance=float(explained_variance),
        relative_reconstruction_rmse=float(relative_rmse),
    )


@lru_cache(maxsize=1)
def reference_sae() -> TopKSparseAutoencoder:
    """Return deterministic root-side SAE weights for all randomized episodes."""
    torch.set_num_threads(1)
    artifact = Path("/opt/causal-feature-lab/sae_state.pt")
    if artifact.is_file():
        sae = TopKSparseAutoencoder()
        state = torch.load(artifact, map_location="cpu", weights_only=True)
        sae.load_state_dict(state)
        sae.eval()
        for parameter in sae.parameters():
            parameter.requires_grad_(False)
        return sae
    sae = fit_sae(
        seed=0x51AE,
        causal_concept=0,
        gain=1.0,
    ).sae
    sae.eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae
