"""
Prototype TopK Sparse Autoencoder (ProtoSAE).

Architecture
------------
Same outer skeleton as a standard TopK SAE (encoder -> sparsifying activation
-> decoder), but the encoder is reinterpreted: each of the n_latents rows of
the encoder weight matrix W is treated as a projection of the LLM embedding
into a *separate* 1D prototype space (one per latent).

For latent j:
    s_j = W[j, :] @ x + b[j]    # scalar 1D projection
    p_j (learned scalar prototype, one per latent)
    d_j^2 = (s_j - p_j)^2

Activation is a monotonically decreasing function of distance, taken from the
original ProtoPNet paper (Chen et al. 2019):

    sim_j = log( (d_j^2 + 1) / (d_j^2 + eps) )

so an exact match (s_j == p_j) gives a large positive activation log(1/eps),
while points far from the prototype have sim -> 0+. TopK is then applied to
the similarities, and the top-k latents are used to reconstruct x via a
standard linear decoder, exactly like the normal SAE.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.topk_sae import TopK  # reuse the existing TopK so behaviour matches


class ProtoAutoencoder(nn.Module):
    """
    Prototype TopK SAE.

    Mirrors the public interface of the standard `Autoencoder` so it can be
    a drop-in replacement in the training / eval loops:
        - forward(x)            -> (lat_pre, lat, recon)
        - encode_pre_act(x)     -> similarities  [B, n_latents]
        - self.pre_bias         -> Parameter [d_model]
        - self.decoder.weight   -> [d_model, n_latents]   (column-norm-able)
        - self.n_latents
    """

    def __init__(
        self,
        n_latents: int,
        d_model: int,
        activation: nn.Module,
        normalize: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.normalize = normalize
        self.eps = eps

        # Subtracted from input, added back to reconstruction.
        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Encoder = n_latents independent 1D projections of x.
        self.encoder = nn.Linear(d_model, n_latents, bias=True)

        # One scalar prototype per latent. Init from a small Gaussian so
        # latents specialise on different parts of their projection axis
        # rather than all collapsing to the same point.
        self.prototypes = nn.Parameter(torch.randn(n_latents) * 0.1)

        # Per-latent learnable scale on the ProtoPNet similarity.
        #
        # Why this is needed: log((d^2 + 1) / (d^2 + eps)) has a fixed peak
        # of log(1/eps) at d=0. With small eps (1e-3 in ProtoPNet), that
        # peak is ~6.9, and TopK reliably picks the latents nearest their
        # prototypes -- so the top-k activations are pinned near the peak.
        # Combined with unit-norm decoder columns, that gives an initial
        # ||recon|| ~ 6.9 * sqrt(k) which can dwarf ||x|| for layernormed
        # LLM activations and either makes loss explode or trips the AMP
        # GradScaler. A per-latent scale lets the optimiser pick the right
        # specificity for each prototype while preserving monotonicity.
        # softplus keeps it strictly positive so similarity stays
        # monotone-decreasing in distance throughout training.
        init_scale = 1.0 / math.log(1.0 / eps)               # peak at d=0 -> 1.0
        init_scale_raw = math.log(math.expm1(init_scale))    # softplus^-1
        self.scale_raw = nn.Parameter(
            torch.full((n_latents,), init_scale_raw)
        )

        # Decoder: same as standard SAE. Bias is handled by pre_bias.
        self.decoder = nn.Linear(n_latents, d_model, bias=False)

        self.activation = activation

    # ------------------------------------------------------------------
    #  Encoding
    # ------------------------------------------------------------------
    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x into n_latents 1D prototype spaces and return the
        ProtoPNet-style similarity to the prototype in each space, with a
        per-latent learnable scale. Higher = closer to the prototype =
        stronger latent activation.
        """
        x = x - self.pre_bias
        s = self.encoder(x)                          # [B, n_latents]
        d_sq = (s - self.prototypes).pow(2)          # [B, n_latents]
        sim_raw = torch.log((d_sq + 1.0) / (d_sq + self.eps))
        scale = F.softplus(self.scale_raw)           # > 0, [n_latents]
        return scale * sim_raw

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        lat_pre = self.encode_pre_act(x)
        lat = self.activation(lat_pre)               # TopK
        recon = self.decoder(lat) + self.pre_bias
        return lat_pre, lat, recon

    # ------------------------------------------------------------------
    #  Hook for the AuxK auxiliary loss
    # ------------------------------------------------------------------
    def aux_dead_activation(self, aux_pre: torch.Tensor) -> torch.Tensor:
        """
        Used by the AuxK auxiliary loss to decode dead-latent
        pre-activations. For ProtoSAE these *are* already monotonic
        similarities, so we just clamp to non-negative (matching the
        intuition that an "active" prototype latent has positive similarity)
        rather than applying torch.exp as the standard SAE does.
        """
        return aux_pre.clamp(min=0.0)

    # ------------------------------------------------------------------
    #  One-batch scale calibration (option B)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_scale_from_batch(self, x: torch.Tensor, k: int) -> tuple:
        """
        Reset the per-latent scale so that, at init, the reconstruction
        magnitude is roughly ||x||.

        Reasoning. With unit-norm decoder columns and the top-k latents
        selected by TopK, all k active activations sit near the peak of
        their similarity curve (= scale * log(1/eps)). The k decoder
        columns are approximately orthogonal in high dimensions, so

            ||recon|| ~ peak * sqrt(k) = scale * log(1/eps) * sqrt(k).

        We want this to match the expected ||x||, so

            scale_target = ||x|| / (log(1/eps) * sqrt(k)).

        We then store this as the inverse-softplus of the raw parameter,
        so softplus(scale_raw) = scale_target. Returns (old, new) for
        logging.
        """
        old_scale = float(F.softplus(self.scale_raw).mean().item())

        x_centered = x - self.pre_bias
        x_norm = float(x_centered.norm(dim=-1).mean().item())
        target_scale = x_norm / (math.log(1.0 / self.eps) * math.sqrt(k))
        # Numerically stable inverse-softplus.
        target_scale_raw = math.log(math.expm1(target_scale))
        self.scale_raw.data.fill_(target_scale_raw)

        return old_scale, target_scale
        
        
        
        
        
# """
# Prototype Sparse Autoencoder (ProtoSAE)

# Replaces the linear encoder with proximity-to-prototype activation.
# Each latent feature is defined by a learned prototype vector in activation
# space. Activation is based on similarity between the input and the prototype,
# making each feature intrinsically interpretable.

# Same interface as topk_sae.Autoencoder: forward() returns (pre_act, latents, recons).
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Any, Callable

# from src.topk_sae import TopK, LN


# # ---------------------------------------------------------------------------
# #                      SIMILARITY FUNCTIONS
# # ---------------------------------------------------------------------------

# class EuclideanSimilarity(nn.Module):
#     """
#     Proximity = 1 - ||x - p|| / r
#     1 at prototype center, 0 at radius boundary, negative outside.
#     Decision boundary: hypersphere.
#     """
#     name = "euclidean"

#     def forward(
#         self, x: torch.Tensor, prototypes: torch.Tensor, radii: torch.Tensor
#     ) -> torch.Tensor:
#         # ||x - p||^2 = ||x||^2 + ||p||^2 - 2 x·p
#         x_sq = (x ** 2).sum(dim=-1, keepdim=True)          # (B, 1)
#         p_sq = (prototypes ** 2).sum(dim=-1).unsqueeze(0)   # (1, N)
#         cross = x @ prototypes.T                             # (B, N)
#         sq_dist = (x_sq + p_sq - 2 * cross).clamp(min=0)
#         dist = sq_dist.sqrt()
#         return 1.0 - dist / (radii.abs() + 1e-8)            # (B, N)


# class CosineSimilarity(nn.Module):
#     """
#     Proximity = cosine_similarity(x, p)
#     Range [-1, 1]. Decision boundary: hypercone.
#     Ignores magnitude — purely directional.
#     """
#     name = "cosine"

#     def forward(
#         self, x: torch.Tensor, prototypes: torch.Tensor, radii: torch.Tensor
#     ) -> torch.Tensor:
#         x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
#         p_norm = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-8)
#         return x_norm @ p_norm.T  # (B, N)


# class RBFSimilarity(nn.Module):
#     """
#     Proximity = exp(-||x - p||^2 / (2 * sigma^2))
#     Range (0, 1]. sigma = radius. Decision boundary: soft hypersphere.
#     The classic radial basis function — universal approximator.
#     """
#     name = "rbf"

#     def forward(
#         self, x: torch.Tensor, prototypes: torch.Tensor, radii: torch.Tensor
#     ) -> torch.Tensor:
#         x_sq = (x ** 2).sum(dim=-1, keepdim=True)
#         p_sq = (prototypes ** 2).sum(dim=-1).unsqueeze(0)
#         cross = x @ prototypes.T
#         sq_dist = (x_sq + p_sq - 2 * cross).clamp(min=0)
#         sigma_sq = (radii.abs() + 1e-8) ** 2
#         return torch.exp(-sq_dist / (2 * sigma_sq))  # (B, N)


# SIMILARITY_CLASSES = {
#     "euclidean": EuclideanSimilarity,
#     "cosine": CosineSimilarity,
#     "rbf": RBFSimilarity,
# }


# # ---------------------------------------------------------------------------
# #                      PROTO AUTOENCODER
# # ---------------------------------------------------------------------------

# class ProtoAutoencoder(nn.Module):
#     """
#     Prototype-based Sparse Autoencoder.

#     Architecture:
#         encode:  z_i = TopK( scale_i * similarity(x, prototype_i) )
#         decode:  x_hat = z @ W_dec + pre_bias

#     Key differences from standard SAE:
#         - Encoder is non-linear (distance/similarity based)
#         - Each feature has a concrete prototype that can be mapped to
#           nearest training examples for free interpretability
#         - Radii/bandwidth parameters control activation region size
#         - Per-feature scale parameters bridge bounded similarity values
#           with the decoder's unit-norm constraint

#     Same parameter count as standard SAE (+ n_latents for radii + scales).
#     """

#     def __init__(
#         self,
#         n_latents: int,
#         n_inputs: int,
#         activation: Callable = nn.ReLU(),
#         normalize: bool = True,
#         similarity: str = "euclidean",  # "euclidean", "cosine", "rbf"
#     ) -> None:
#         super().__init__()

#         self.n_latents = n_latents
#         self.n_inputs = n_inputs
#         self.activation = activation
#         self.normalize = normalize
#         self.similarity_name = similarity

#         # Similarity function
#         if similarity not in SIMILARITY_CLASSES:
#             raise ValueError(
#                 f"Unknown similarity '{similarity}'. "
#                 f"Choose from: {list(SIMILARITY_CLASSES.keys())}"
#             )
#         self.similarity_fn = SIMILARITY_CLASSES[similarity]()

#         # Prototypes: live in activation space (same as encoder weights)
#         self.prototypes = nn.Parameter(torch.empty(n_latents, n_inputs))

#         # Per-feature radius (for euclidean/rbf) or unused placeholder (cosine)
#         self.radii = nn.Parameter(torch.ones(n_latents))

#         # Per-feature scale: maps bounded similarity → reconstruction-scale
#         # magnitude. Keeps decoder unit-norm stable.
#         self.scales = nn.Parameter(torch.ones(n_latents))

#         # Bias (same role as pre_bias in standard SAE)
#         self.pre_bias = nn.Parameter(torch.zeros(n_inputs))

#         # Decoder: same as standard SAE (linear, unit-norm rows)
#         self.decoder_weight = nn.Parameter(torch.empty(n_latents, n_inputs))

#         # Stats buffers (compatibility with training loop)
#         self.register_buffer(
#             "stats_last_nonzero",
#             torch.zeros(n_latents, dtype=torch.long),
#         )
#         self.register_buffer(
#             "latents_activation_frequency",
#             torch.ones(n_latents, dtype=torch.float),
#         )
#         self.register_buffer(
#             "latents_mean_square",
#             torch.zeros(n_latents, dtype=torch.float),
#         )

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.normal_(self.prototypes, std=1.0)
#         nn.init.kaiming_uniform_(self.decoder_weight)
#         # Normalize decoder rows to unit norm
#         with torch.no_grad():
#             self.decoder_weight.data = F.normalize(
#                 self.decoder_weight.data, dim=-1
#             )

#     @torch.no_grad()
#     def init_from_data(self, data: torch.Tensor):
#         """
#         Initialize prototypes from real activation data. Call once before
#         training on the first chunk/batch.

#         - Samples n_latents points from data (with replacement + noise)
#         - Sets radii from median nearest-neighbor distance
#         - Sets scales so initial reconstruction is in the right ballpark
#         - Aligns decoder directions with prototype directions
#         """
#         n = data.shape[0]

#         # Preprocess if needed (match what encode sees)
#         if self.normalize:
#             data_proc, _ = self.preprocess(data)
#         else:
#             data_proc = data

#         # Sample prototypes from data
#         indices = torch.randint(0, n, (self.n_latents,))
#         self.prototypes.data = data_proc[indices].clone()
#         self.prototypes.data += torch.randn_like(self.prototypes.data) * 0.01

#         # Estimate radii from data scale
#         sample = data_proc[: min(2000, n)]
#         dists = torch.cdist(sample, sample)
#         dists.fill_diagonal_(float("inf"))
#         nn_dists = dists.min(dim=1).values
#         median_nn = nn_dists.median().item()

#         if self.similarity_name == "euclidean":
#             self.radii.data.fill_(median_nn * 3.0)
#         elif self.similarity_name == "rbf":
#             self.radii.data.fill_(median_nn * 1.5)  # sigma
#         else:
#             self.radii.data.fill_(1.0)  # unused for cosine

#         # Set scales: want k * scale * ~0.5 ≈ typical activation norm
#         # After LN, data_proc has std≈1, norm≈sqrt(d)
#         data_norm = data_proc.norm(dim=-1).mean().item()
#         k = self.activation.k if hasattr(self.activation, "k") else 64
#         init_scale = data_norm / (k * 0.5)
#         self.scales.data.fill_(init_scale)

#         # Decoder directions from prototypes
#         self.decoder_weight.data = F.normalize(self.prototypes.data.clone(), dim=-1)

#         print(
#             f"  ProtoSAE init: {self.n_latents} prototypes, "
#             f"similarity={self.similarity_name}, "
#             f"median_nn={median_nn:.2f}, "
#             f"radius={self.radii[0].item():.2f}, "
#             f"scale={init_scale:.2f}"
#         )

#     # -- Preprocessing (same as standard SAE) ---------------------------------

#     def preprocess(
#         self, x: torch.Tensor
#     ) -> tuple[torch.Tensor, dict[str, Any]]:
#         if not self.normalize:
#             return x, {}
#         x, mu, std = LN(x)
#         return x, {"mu": mu, "std": std}

#     # -- Encode ---------------------------------------------------------------

#     def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
#         """Compute scaled similarity to all prototypes (before activation)."""
#         x = x - self.pre_bias
#         similarity = self.similarity_fn(x, self.prototypes, self.radii)  # (B, N)
#         return self.scales.abs() * similarity  # (B, N)

#     def encode(
#         self, x: torch.Tensor
#     ) -> tuple[torch.Tensor, dict[str, Any]]:
#         x, info = self.preprocess(x)
#         return self.activation(self.encode_pre_act(x)), info

#     # -- Decode ---------------------------------------------------------------

#     def decode(
#         self, latents: torch.Tensor, info: dict[str, Any] | None = None
#     ) -> torch.Tensor:
#         ret = F.linear(latents, self.decoder_weight.T) + self.pre_bias
#         if self.normalize and info is not None:
#             ret = ret * info["std"] + info["mu"]
#         return ret

#     # -- Forward (same interface as Autoencoder) ------------------------------

#     def forward(
#         self, x: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         x, info = self.preprocess(x)
#         latents_pre_act = self.encode_pre_act(x)
#         latents = self.activation(latents_pre_act)
#         recons = self.decode(latents, info)

#         # Update dead-feature stats
#         self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
#         self.stats_last_nonzero += 1

#         return latents_pre_act, latents, recons

#     # -- Decoder normalization ------------------------------------------------

#     @torch.no_grad()
#     def normalize_decoder(self):
#         """Constrain decoder rows to unit norm."""
#         self.decoder_weight.data = F.normalize(
#             self.decoder_weight.data, dim=-1
#         )

#     # -- Interpretability: nearest neighbors ----------------------------------

#     @torch.no_grad()
#     def get_prototype_nearest_neighbors(
#         self,
#         data: torch.Tensor,
#         top_k: int = 10,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         For each prototype, find the top_k nearest training examples.

#         This is the core interpretability advantage: no need to run the
#         full dataset through the encoder. Just do nearest-neighbor lookup
#         in activation space.

#         Returns:
#             indices: (n_latents, top_k) indices into data
#             distances: (n_latents, top_k) Euclidean distances
#         """
#         if self.normalize:
#             data, _ = self.preprocess(data)

#         chunk_size = 4096
#         device = self.prototypes.device

#         all_min_dists = torch.full(
#             (self.n_latents, top_k), float("inf"), device=device
#         )
#         all_min_indices = torch.zeros(
#             (self.n_latents, top_k), dtype=torch.long, device=device
#         )

#         for start in range(0, len(data), chunk_size):
#             chunk = data[start : start + chunk_size].to(device)
#             dists = torch.cdist(self.prototypes, chunk)  # (N, chunk)

#             for i in range(self.n_latents):
#                 combined_dists = torch.cat([all_min_dists[i], dists[i]])
#                 combined_indices = torch.cat(
#                     [
#                         all_min_indices[i],
#                         torch.arange(
#                             start, start + len(chunk), device=device
#                         ),
#                     ]
#                 )
#                 topk_vals, topk_idx = combined_dists.topk(
#                     top_k, largest=False
#                 )
#                 all_min_dists[i] = topk_vals
#                 all_min_indices[i] = combined_indices[topk_idx]

#         return all_min_indices, all_min_dists

#     # -- Serialization --------------------------------------------------------

#     def state_dict(self, destination=None, prefix="", keep_vars=False):
#         sd = super().state_dict(destination, prefix, keep_vars)
#         sd[prefix + "similarity"] = self.similarity_name
#         sd[prefix + "activation"] = self.activation.__class__.__name__
#         if hasattr(self.activation, "state_dict"):
#             sd[prefix + "activation_state_dict"] = self.activation.state_dict()
#         return sd

#     @classmethod
#     def from_state_dict(
#         cls, state_dict: dict, strict: bool = True
#     ) -> "ProtoAutoencoder":
#         n_latents, n_inputs = state_dict["prototypes"].shape
#         similarity = state_dict.pop("similarity", "euclidean")
#         activation_name = state_dict.pop("activation", "TopK")

#         from src.topk_sae import ACTIVATIONS_CLASSES
#         activation_class = ACTIVATIONS_CLASSES.get(activation_name, nn.ReLU)
#         activation_sd = state_dict.pop("activation_state_dict", {})
#         if hasattr(activation_class, "from_state_dict"):
#             activation = activation_class.from_state_dict(activation_sd)
#         else:
#             activation = activation_class()

#         normalize = activation_name == "TopK"
#         model = cls(
#             n_latents, n_inputs,
#             activation=activation,
#             normalize=normalize,
#             similarity=similarity,
#         )
#         model.load_state_dict(state_dict, strict=strict)
#         return model
