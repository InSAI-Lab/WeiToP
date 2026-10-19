"""WeiAD aggregation and WeiToP token pruning for visual place recognition."""

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def log_optimal_transport(scores: Tensor, dustbin_score: Tensor) -> Tensor:
    """Return a rescaled log transport plan of shape [B, M + 1, N + 1].

    Rows represent clusters and columns represent patch tokens. The last row
    is the dustbin cluster; the last column is the ghost token. Both use the
    same learnable affinity.
    """
    batch_size, num_clusters, num_tokens = scores.shape
    # Keep Sinkhorn arithmetic in float32 when the projections use mixed precision.
    scores = scores.float()
    dustbin_score = dustbin_score.float()
    augmented = torch.cat(
        [scores, dustbin_score.expand(batch_size, num_clusters, 1)], dim=2
    )
    augmented = torch.cat(
        [augmented, dustbin_score.expand(batch_size, 1, num_tokens + 1)], dim=1
    )

    # SuperGlue marginals, in cluster-by-token order. Rescaling below restores
    # unit mass for each real cluster/token at convergence.
    log_norm = -math.log(num_clusters + num_tokens)
    log_a = scores.new_full((batch_size, num_clusters + 1), log_norm)
    log_b = scores.new_full((batch_size, num_tokens + 1), log_norm)
    log_a[:, -1] += math.log(num_tokens)
    log_b[:, -1] += math.log(num_clusters)
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    # Use three Sinkhorn iterations with entropic regularisation epsilon = 1.
    for _ in range(3):
        u = log_a - torch.logsumexp(augmented + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(augmented + u.unsqueeze(2), dim=1)
    return augmented + u.unsqueeze(2) + v.unsqueeze(1) - log_norm


class WeiAD(nn.Module):
    """Aggregate final DINOv2 patch and [CLS] features into an 8448-D descriptor.

    This is the aggregation head. Use WeiToP(keep_ratio=1.0) for the complete
    unpruned image encoder. forward returns (descriptor, token_importance).
    """

    num_channels = 768
    num_clusters = 64
    cluster_dim = 128
    token_dim = 256
    descriptor_dim = num_clusters * cluster_dim + token_dim
    group_sizes = (24, 20, 16, 4)

    def __init__(self) -> None:
        super().__init__()
        self.token_features = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 256)
        )
        self.cluster_features = nn.Sequential(
            nn.Linear(768, 512), nn.Dropout(0.3), nn.ReLU(), nn.Linear(512, 128)
        )
        self.score = nn.Sequential(
            nn.Linear(768, 512), nn.Dropout(0.3), nn.ReLU(), nn.Linear(512, 64)
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        # Initialise tiers with uniform unconstrained weight decrements.
        # These give initial weights approximately [1, 0.687, 0.373, 0.060].
        self.tier_base = nn.Parameter(torch.zeros(1))
        self.tier_delta = nn.Parameter(torch.full((3,), -1.0))

    def get_tier_weights(self) -> Tensor:
        """Positive, non-increasing weights for the four rank-based tiers."""
        base = self.tier_base.exp()
        decrements = F.softplus(self.tier_delta).cumsum(dim=0)
        return torch.cat([base, base - decrements]).clamp_min(1e-3)

    def cluster_weights(self, transport: Tensor) -> Tensor:
        """Assign tier weights whilst preserving cluster indices."""
        real_mass = transport[:, :-1, :-1].sum(dim=-1)
        ghost_mass = transport[:, :-1, -1]
        ranks = (real_mass - 0.5 * ghost_mass).argsort(dim=1, descending=True)
        ranked_weights = torch.repeat_interleave(
            self.get_tier_weights(),
            torch.tensor(self.group_sizes, device=transport.device),
        ).to(transport.dtype)
        # Ranking is discrete; gradients still reach the selected tier weights.
        return torch.zeros_like(real_mass).scatter(
            1, ranks, ranked_weights.unsqueeze(0).expand_as(real_mass)
        )

    def forward(self, patches: Tensor, cls_token: Tensor) -> tuple[Tensor, Tensor]:
        """Accept patches [B, N, 768] and a [CLS] token [B, 768]."""
        local_features = self.cluster_features(patches).transpose(1, 2)
        scores = self.score(patches).transpose(1, 2)
        transport = log_optimal_transport(scores, self.dust_bin).exp()
        weights = self.cluster_weights(transport)
        weighted_transport = transport[:, :-1, :-1] * weights.unsqueeze(-1)

        # Reuse this plan for aggregation and teacher supervision. Matrix
        # multiplication avoids materialising [B, 128, 64, N] intermediates.
        with torch.autocast(device_type=patches.device.type, enabled=False):
            local_descriptor = torch.bmm(
                local_features.float(), weighted_transport.transpose(1, 2)
            )
        cls_descriptor = self.token_features(cls_token).float()

        # Concatenate weighted cluster sums and projected [CLS], then apply
        # global normalisation to preserve relative cluster contributions.
        # Flatten local descriptors in [feature, cluster] order.
        descriptor = F.normalize(
            torch.cat([cls_descriptor, local_descriptor.flatten(1)], dim=-1),
            p=2, dim=-1,
        )
        importance = weighted_transport.sum(dim=1)
        return descriptor, importance


class WeiToP(nn.Module):
    """DINOv2 ViT-B/14 + WeiAD + the early WeiToP importance predictor.

    Training retains all patch tokens and returns (descriptor, KL loss).
    Evaluation uses keep_ratio and returns (descriptor, None). All ratios
    share the same parameters; no additional training is needed.
    """

    descriptor_dim = WeiAD.descriptor_dim

    def __init__(
        self,
        keep_ratio: float = 1.0,
        *,
        pretrained: bool = True,
        dinov2_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.alpha = 0.5
        self.tau = 0.1
        if dinov2_path is None:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained
            )
        else:
            # A local DINOv2 checkout permits offline architecture loading.
            self.backbone = torch.hub.load(
                str(Path(dinov2_path).expanduser().resolve()),
                "dinov2_vitb14",
                source="local",
                pretrained=pretrained,
            )
        self.aggregation = WeiAD()
        self.prune_student = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        # Freeze the embedding stem and blocks 1--8. Fine-tune the last four
        # transformer blocks and the final LayerNorm.
        self.backbone.requires_grad_(False)
        self.backbone.blocks[-4:].requires_grad_(True)
        self.backbone.norm.requires_grad_(True)

    @property
    def keep_ratio(self) -> float:
        return self._keep_ratio

    @keep_ratio.setter
    def keep_ratio(self, value: float) -> None:
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("keep_ratio must be finite and in (0, 1].")
        self._keep_ratio = value

    def prune_tokens(self, x: Tensor, logits: Tensor) -> Tensor:
        """Rank patches by predicted importance and magnitude; retain [CLS]."""
        logits_min = logits.amin(dim=1, keepdim=True)
        logits_max = logits.amax(dim=1, keepdim=True)
        logits_norm = (logits - logits_min) / (logits_max - logits_min + 1e-8)
        scores = self.alpha * logits_norm + (1 - self.alpha) * torch.norm(
            x[:, 1:, :], p=2, dim=-1
        )
        # Retain at least one patch token for every positive retention ratio.
        num_kept = math.ceil(self.keep_ratio * scores.shape[1])
        indices = scores.topk(num_kept, dim=1).indices
        patches = x[:, 1:, :].gather(
            1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        return torch.cat([x[:, :1, :], patches], dim=1)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor | None]:
        """Encode ImageNet-normalised RGB images [B, 3, H, W].

        Both spatial dimensions must be positive multiples of 14. The
        descriptor has unit L2 norm and dimension 8448 at every keep_ratio.
        """
        if (
            images.ndim != 4
            or images.shape[0] == 0
            or images.shape[1] != 3
            or any(size <= 0 or size % 14 for size in images.shape[-2:])
        ):
            raise ValueError("Expected [B, 3, H, W] with positive H and W divisible by 14.")
        with torch.no_grad():
            x = self.backbone.prepare_tokens_with_masks(images)
            x = self.backbone.blocks[0](x)

        # The predictor learns from final aggregation, without updating the
        # frozen early backbone. Float32 avoids min--max/KL underflow in AMP.
        with torch.autocast(device_type=images.device.type, enabled=False):
            logits = self.prune_student(x[:, 1:, :].float()).squeeze(-1)
            if not self.training and self.keep_ratio < 1.0:
                x = self.prune_tokens(x.float(), logits)

        with torch.no_grad():
            for block in self.backbone.blocks[1:-4]:
                x = block(x)
        for block in self.backbone.blocks[-4:]:
            x = block(x)
        x = self.backbone.norm(x)
        descriptor, importance = self.aggregation(x[:, 1:, :], x[:, 0, :])

        loss_distill = None
        if self.training:
            # Use detached log-importance as the teacher and average the
            # token-wise KL divergence over images in the batch.
            with torch.no_grad():
                teacher_logits = (importance.detach() + 1e-8).log()
                teacher_probs = F.softmax(teacher_logits / self.tau, dim=-1)
            student_log_probs = F.log_softmax(logits / self.tau, dim=-1)
            loss_distill = F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean"
            ) * self.tau**2
        return descriptor, loss_distill
