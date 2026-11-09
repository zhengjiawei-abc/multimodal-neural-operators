from __future__ import annotations

"""1-D Burgers hybrid PINN, with opt-in residual-convolution variants.

Copy of ``demo_burgers_dataset.py`` (2026-10-25) that adds four defaults-off
CLI switches so the ResNet question can be tested without touching the main
training script:

  --cnn-res-blocks N / --cnn-width W
      input-tokenizer CNN becomes: conv stem -> N x ConvResBlock -> 1x1 mixer
  --manifold-cnn-res N / --manifold-cnn-width W
      the D7 latent->CNN branch becomes: conv stem -> N x ConvResBlock -> 1x1

Everything else is identical to demo_burgers_dataset.py; with all four switches
unset the constructed model and the initialisation RNG order are unchanged.
"""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

# Must be set before CUDA is initialised. The default caching allocator grows a
# fragmented reserved pool over long runs, and on Windows every reserved byte
# also raises host commit charge through WDDM - a 12000-step run was seen
# holding ~7.4 GB of device memory and ~9.4 GB of commit for a 300k-parameter
# model. Expandable segments keeps that pool from fragmenting, but the backend
# is not implemented on Windows (it only prints a warning there), so it is
# enabled on other platforms only. The portable part of the fix is the
# periodic ``torch.cuda.empty_cache()`` after each evaluation.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DatasetConfig:
    data_dir: str = "datasets/pdebench_burgers"
    train_file: str = "burgers_nu0.01_train_000.parquet"
    dev_file: str = "burgers_nu0.01_dev.parquet"
    test_file: str = "burgers_nu0.01_test.parquet"
    train_samples: int = 512
    dev_samples: int = 64
    test_samples: int = 128
    eval_samples: int = 32
    nx: int = 256
    nt: int = 101
    nu: float = 0.01
    latent_dim: int = 16
    manifold_hidden: int = 128
    manifold_arch: str = "mlp"
    conv_ae_channels: int = 16
    cnn_res_blocks: int = 0
    cnn_width: int = 16
    manifold_cnn_res_blocks: int = 0
    manifold_cnn_width: int = 0
    distill_fno_path: str = ""
    distill_weight: float = 0.0
    distill_points_per_sample: int = 0
    denoise_sigma: float = 0.0
    denoise_weight: float = 0.0
    denoise_steps: int = 0
    denoise_rate: float = 0.5
    denoise_eval_sigma: float = 0.0
    denoise_until: int = -1
    embed_flow: bool = False
    embed_flow_max_sigma: float = 1.0
    embed_flow_weight: float = 0.1
    use_cross_attention: bool = True
    tokenizer: str = "cnn"
    arch: str = "full"
    dropout: float = 0.0
    fourier_bands: int = 4
    time_fourier_bands: int = 4
    cnn_channels: int = 24
    token_count: int = 16
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    head_width: int = 32
    batch_size: int = 64
    data_points_per_sample: int = 32
    physics_points_per_sample: int = 4
    initial_points_per_sample: int = 16
    boundary_points_per_sample: int = 1
    initial_patch_radius: int = 0
    physics_slope_bias: float = 0.0
    periodic_derivative_weight: float = 0.0
    use_manifold_cnn: bool = False
    query_local_features: bool = False
    query_local_window: int = 0
    time_conditioned_context: bool = False
    data_gradient_weight: float = 0.0
    decoder: str = "query"
    grid_channels: int = 16
    grid_layers: int = 3
    grid_seed: int = 16
    spectral_modes_x: int = 12
    spectral_modes_t: int = 8
    latent_ode: bool = False
    latent_channels: int = 16
    latent_levels: int = 3
    ode_steps: int = 4
    ode_hidden: int = 64
    temporal: str = "ode"
    rollout_steps: int = 8
    rollout_teacher_forcing: float = 0.0
    rollout_delta: bool = False
    operator_attention: bool = False
    manifold_steps: int = 500
    steps: int = 6000
    lr_decay_steps: int = 4000
    lr: float = 5.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    min_lr_ratio: float = 0.02
    loss_balance: bool = True
    loss_floor: float = 1.0e-3
    loss_balance_warmup: int = 0
    warmup_steps: int = 200
    physics_weight: float = 0.05
    validation_physics_weight: float = 0.1
    physics_nonlinear_scale: float = 1.0
    eval_every: int = 200
    early_stop_patience: int = 8
    early_stop_min_delta: float = 0.0
    log_every: int = 50
    seed: int = 23


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def configure_backend(device: torch.device) -> None:
    """Enable the safe fp32 speedups: TF32 tensor cores and cuDNN autotune."""
    if device.type != "cuda":
        return
    # TF32 keeps fp32 storage but rounds the multiply-accumulate, which is a
    # large win on Ada/Ampere tensor cores with negligible accuracy impact.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Let cuDNN autotune the convolutional manifold / tokenizer kernels.
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in model.named_children()
    }


def clip_gradients(
    model: nn.Module,
    max_norm: float,
) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    )
    return float(grad_norm.detach())


class LossScaleBalancer:
    # Set once from the CLI in ``main``; a class attribute keeps the three
    # training loops from needing ``cfg`` threaded through their constructors.
    default_warmup: int = 0

    def __init__(
        self,
        enabled: bool,
        floor: float = 1.0e-3,
        warmup: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.floor = floor
        self.warmup = self.default_warmup if warmup is None else warmup
        self.step = 0
        self.references: Dict[str, float] = {}

    def set_step(self, step: int) -> None:
        self.step = step

    def scale(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not torch.isfinite(loss.detach()):
            return loss
        value = float(loss.detach())
        if name != "data" and self.step < self.warmup:
            # While the model output is still degenerate an untrained field is
            # close to zero, so its derivatives - and with them the PDE
            # residual - vanish artificially. Freezing a reference here would
            # pin that term to a meaningless scale for the rest of training.
            # The data term stays at full strength so it can drive the early
            # learning; every other term is muted until the output has
            # structure and its reference becomes meaningful.
            return loss * min(1.0, self.floor / max(value, self.floor))
        reference = self.references.get(name)
        if reference is None:
            reference = max(value, self.floor)
            self.references[name] = reference
        multiplier = min(1.0, reference / max(value, self.floor))
        return loss * multiplier


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DatasetConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    decay_horizon = (
        cfg.lr_decay_steps
        if cfg.lr_decay_steps > 0
        else cfg.steps
    )
    warmup_steps = min(cfg.warmup_steps, cfg.steps, decay_horizon)
    decay_steps = max(decay_horizon - warmup_steps, 1)

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            warmup_progress = (step + 1) / warmup_steps
            return 0.1 + 0.9 * warmup_progress
        progress = min(
            1.0,
            max(0.0, (step - warmup_steps) / decay_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = cfg.lr * lr_factor(0)
    return scheduler


class Fourier2D(nn.Module):
    def __init__(self, bands: int = 4, time_bands: int = 4) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(bands).float() * math.pi)
        self.register_buffer(
            "time_freqs",
            2.0 ** torch.arange(time_bands).float() * math.pi,
        )
        self.out_dim = 2 * bands + 2 * time_bands

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xf = x * self.freqs.unsqueeze(0)
        tf = t * self.time_freqs.unsqueeze(0)
        return torch.cat(
            [torch.sin(xf), torch.cos(xf), torch.sin(tf), torch.cos(tf)],
            dim=-1,
        )


class TimeFourier(nn.Module):
    def __init__(self, bands: int = 4) -> None:
        super().__init__()
        self.register_buffer(
            "freqs",
            2.0 ** torch.arange(bands).float() * math.pi,
        )
        self.out_dim = 2 * bands

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tf = t * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(tf), torch.cos(tf)], dim=-1)


def interpolate_initial_field(
    field: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    position = x.squeeze(-1).clamp(0.0, 1.0) * (field.size(1) - 1)
    left = torch.floor(position).long().clamp(0, field.size(1) - 2)
    fraction = (position - left.float()).unsqueeze(1)
    left_value = field.gather(1, left.unsqueeze(1))
    right_value = field.gather(1, (left + 1).unsqueeze(1))
    return left_value + fraction * (right_value - left_value)


def interpolate_initial_patch(
    field: torch.Tensor,
    x: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    spacing = 1.0 / (field.size(1) - 1)
    values = []
    for offset in range(-radius, radius + 1):
        shifted_x = torch.remainder(x + offset * spacing, 1.0)
        values.append(interpolate_initial_field(field, shifted_x))
    return torch.cat(values, dim=-1)


def interpolate_feature_map(
    features: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    channel_count = features.size(1)
    grid = features.size(2)
    position = x.squeeze(-1).clamp(0.0, 1.0) * (grid - 1)
    left = torch.floor(position).long().clamp(0, grid - 2)
    fraction = (position - left.float()).view(-1, 1, 1)
    left_index = left.view(-1, 1, 1).expand(-1, channel_count, 1)
    right_index = (left + 1).view(-1, 1, 1).expand(-1, channel_count, 1)
    left_value = features.gather(2, left_index)
    right_value = features.gather(2, right_index)
    return (left_value + fraction * (right_value - left_value)).squeeze(-1)


def repeated_feature_source(
    initial_field: torch.Tensor,
    x: torch.Tensor,
    batch_size: int,
) -> "Tuple[torch.Tensor, int] | None":
    """Detect a ``repeat_interleave`` layout in the query field.

    Returns ``(unique_rows, repeat)`` when ``initial_field`` holds one row per
    trajectory repeated ``repeat`` times, and ``None`` when the rows are not a
    plain repeat.  The equality check keeps the optimisation numerically safe:
    a caller whose rows happen to divide evenly but differ falls back to the
    exact per-row path.
    """

    count = initial_field.size(0)
    if (
        batch_size <= 0
        or count <= batch_size
        or count % batch_size != 0
        or x.size(0) != count
    ):
        return None
    repeat = count // batch_size
    grouped = initial_field.reshape(
        batch_size,
        repeat,
        initial_field.size(1),
    )
    if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
        return None
    return grouped[:, 0], repeat


def query_local_features(
    model: BurgersHybridPINN,
    initial_field: torch.Tensor,
    x: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Local tokenizer features at query points.

    Callers usually expand one field row per trajectory into one row per query
    point before calling this helper.  Running the tokenizer on the expanded
    tensor materialises a ``[batch * points, channels, nx]`` feature map whose
    autograd tape dominated device memory and tripped CUDA OOM on an 8 GB card,
    so a repeated layout is detected and the map is computed once per unique
    row instead.
    """

    patch = interpolate_initial_patch(initial_field, x, radius)
    if not model.cfg.query_local_features:
        return patch
    cfg = model.cfg
    window = cfg.query_local_window
    spacing = 1.0 / (cfg.nx - 1)
    offsets = [0] if window <= 0 else list(range(-window, window + 1))
    source = repeated_feature_source(initial_field, x, cfg.batch_size)
    if source is not None:
        features = model.local_feature_map(source[0])
        channels = features.size(1)
        grid = features.size(2)
        flat = features.reshape(features.size(0) * channels, grid)
        row = (
            torch.arange(source[0].size(0), device=x.device)
            .repeat_interleave(source[1])
            .unsqueeze(1)
            * channels
            + torch.arange(channels, device=x.device).unsqueeze(0)
        )
        sampled = []
        for offset in offsets:
            shifted = torch.remainder(x + offset * spacing, 1.0)
            position = shifted.squeeze(-1).clamp(0.0, 1.0) * (grid - 1)
            left = torch.floor(position).long().clamp(0, grid - 2)
            fraction = (position - left.float()).unsqueeze(1)
            left_value = flat[row, left.unsqueeze(1)]
            right_value = flat[row, (left + 1).unsqueeze(1)]
            sampled.append(
                left_value + fraction * (right_value - left_value)
            )
        return torch.cat([patch] + sampled, dim=-1)
    features = model.local_feature_map(initial_field)
    if window <= 0:
        sampled = [interpolate_feature_map(features, x)]
    else:
        sampled = [
            interpolate_feature_map(
                features,
                torch.remainder(x + offset * spacing, 1.0),
            )
            for offset in range(-window, window + 1)
        ]
    return torch.cat([patch] + sampled, dim=-1)


class ManualAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(
            batch,
            length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query = self._split(self.q_proj(query))
        key = self._split(self.k_proj(key))
        value = self._split(self.v_proj(value))
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        batch, _, length, _ = attended.shape
        attended = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out_proj(attended)


class ConvResBlock(nn.Module):
    """Two convolution layers with an identity skip (a small 1-D ResNet block)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout1d(dropout))
        layers.append(nn.SiLU())
        self.block = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.block(features)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = ManualAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized)
        return tokens + self.ffn(self.norm2(tokens))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, 2 * d_model)
                for _ in range(n_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class ManifoldAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class ConvManifoldAutoencoder(nn.Module):
    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.levels = 5
        base = cfg.conv_ae_channels
        channels = [
            base * min(2**level, 4) for level in range(self.levels)
        ]
        encoder_layers: List[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            encoder_layers.append(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                )
            )
            encoder_layers.append(nn.SiLU())
            input_channels = output_channels
        self.encoder = nn.Sequential(*encoder_layers)
        reduced = cfg.nx
        for _ in range(self.levels):
            reduced = max(1, reduced // 2)
        self.reduced = reduced
        self.feature_channels = input_channels
        self.encoded_dim = self.feature_channels * reduced
        self.to_latent = nn.Linear(self.encoded_dim, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, self.encoded_dim)
        decoder_layers: List[nn.Module] = []
        for level in reversed(range(self.levels)):
            output_channels = base * min(2 ** max(level - 1, 0), 4)
            decoder_layers.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            decoder_layers.append(nn.SiLU())
            input_channels = output_channels
        decoder_layers.append(nn.Conv1d(input_channels, 1, kernel_size=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features.unsqueeze(1))
        return self.to_latent(encoded.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        encoded = self.from_latent(latent).view(
            latent.size(0),
            self.feature_channels,
            self.reduced,
        )
        output = self.decoder(encoded)
        if output.size(-1) != self.cfg.nx:
            output = F.interpolate(
                output,
                size=self.cfg.nx,
                mode="linear",
                align_corners=False,
            )
        return output.squeeze(1)


def build_manifold(cfg: DatasetConfig) -> nn.Module:
    if cfg.manifold_arch == "conv":
        return ConvManifoldAutoencoder(cfg)
    return ManifoldAutoencoder(
        cfg.nx,
        cfg.latent_dim,
        cfg.manifold_hidden,
    )


class LatentGridEncoder(nn.Module):
    """u0 [B, nx] -> structured latent grid [B, C, L] (no flattening)."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = 1
        for level in range(cfg.latent_levels):
            out_channels = cfg.latent_channels * min(2**level, 4)
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                )
            )
            layers.append(nn.SiLU())
            in_channels = out_channels
        self.body = nn.Sequential(*layers)
        self.channels = in_channels
        reduced = cfg.nx
        for _ in range(cfg.latent_levels):
            reduced = max(1, reduced // 2)
        self.length = reduced

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        return self.body(u0.unsqueeze(1))


class GridVectorField(nn.Module):
    """Spatially local latent flow: convolution over the latent grid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.time_proj = nn.Linear(3, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        grid: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.time_proj(phase)[:, :, None]
        hidden = F.silu(self.conv1(grid) + bias)
        return self.conv2(hidden)


class StructuredLatentAutoencoder(nn.Module):
    """Grid encoder plus a reconstruction head, so the latent stays faithful."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.encoder = LatentGridEncoder(cfg)
        channels = self.encoder.channels
        self.decoder = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=3, padding=1),
        )
        self.cfg = cfg

    def encode(self, u0: torch.Tensor) -> torch.Tensor:
        return self.encoder(u0)

    def decode(self, grid: torch.Tensor) -> torch.Tensor:
        restored = self.decoder(grid)
        if restored.size(-1) != self.cfg.nx:
            restored = F.interpolate(
                restored,
                size=self.cfg.nx,
                mode="linear",
                align_corners=False,
            )
        return restored.squeeze(1)


class BurgersStructuredLatentPINN(nn.Module):
    """Structured-latent core: grid encoder -> grid flow -> coordinate decoder.

    No autoencoder bottleneck, no tokenizer, no attention: the latent grid is
    read by interpolating it at the query coordinate, so spatial locality comes
    from the latent's own structure.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.manifold = StructuredLatentAutoencoder(cfg)
        self.encoder = self.manifold.encoder
        self.flow = GridVectorField(self.encoder.channels)
        self.fourier = Fourier2D(
            bands=cfg.fourier_bands,
            time_bands=cfg.time_fourier_bands,
        )
        self.query_proj = nn.Sequential(
            nn.Linear(
                self.fourier.out_dim + self.encoder.channels,
                cfg.d_model,
            ),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_width),
            nn.SiLU(),
            nn.Linear(cfg.head_width, 1),
        )
        self.eps_head = None

    def integrate(self, grid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        steps = max(1, self.cfg.ode_steps)
        state = grid
        for index in range(steps):
            phase = time_phase((index + 1) / steps, grid)
            state = state + (t / steps)[:, :, None] * self.flow(state, phase)
        return state

    def sample_grid(
        self,
        grid: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        length = grid.size(2)
        position = x.squeeze(-1).clamp(0.0, 1.0) * (length - 1)
        left = torch.floor(position).long().clamp(0, length - 2)
        fraction = (position - left.float()).view(-1, 1, 1)
        left_index = left.view(-1, 1, 1).expand(-1, grid.size(1), 1)
        right_index = (left + 1).view(-1, 1, 1).expand(-1, grid.size(1), 1)
        left_value = grid.gather(2, left_index)
        right_value = grid.gather(2, right_index)
        return (
            left_value + fraction * (right_value - left_value)
        ).squeeze(-1)

    def encode(
        self,
        u0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = torch.zeros(
            u0.size(0),
            1,
            1,
            device=u0.device,
            dtype=u0.dtype,
        )
        return self.manifold.encode(u0), context

    def decode_queries(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        initial_value: torch.Tensor,
    ) -> torch.Tensor:
        if latent.size(0) != x.size(0):
            latent = expand_latent(latent, x.size(0))
        sampled = self.sample_grid(self.integrate(latent, t), x)
        query = self.query_proj(
            torch.cat([self.fourier(x, t), sampled], dim=-1)
        )
        return self.head(query)

    def corrupt_latent(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor | float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=latent.device)
        return latent, zero, zero

    def prepare_eval_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


def expand_latent(latent: torch.Tensor, count: int) -> torch.Tensor:
    if latent.size(0) == count:
        return latent
    sizes = [count] + [-1] * (latent.dim() - 1)
    return latent.expand(*sizes)


class LatentVectorField(nn.Module):
    def __init__(self, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def time_phase(level: float, reference: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        [
            level,
            math.sin(math.pi * level),
            math.cos(math.pi * level),
        ],
        device=reference.device,
        dtype=reference.dtype,
    )
    return values.view(1, -1).expand(reference.size(0), -1)


class GatedRecurrentTemporal(nn.Module):
    def __init__(self, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(3, latent_dim)

    def forward(self, state: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.cell(phase, state)


class SelectiveStateSpace(nn.Module):
    def __init__(self, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.to_gates = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, state: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        delta, value = self.to_gates(
            torch.cat([state, phase], dim=-1)
        ).chunk(2, dim=-1)
        decay = torch.exp(-F.softplus(delta))
        return decay * state + (1.0 - decay) * torch.tanh(value)


class GridFieldDecoder(nn.Module):
    """FiLM-conditioned field decoder; optionally with residual blocks.

    ``residual=True`` (``--decoder resnet_grid``) turns each FiLM-modulated
    convolution into a residual update, which gives the decoder a much easier
    path to refine sharp fronts without having to relearn the smooth part of
    the field at every layer.
    """

    def __init__(self, cfg: DatasetConfig, residual: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.residual = residual
        channels = cfg.grid_channels
        self.seed = nn.Parameter(
            0.1 * torch.randn(1, channels, cfg.grid_seed, cfg.grid_seed)
        )
        self.seed_film = nn.Linear(cfg.latent_dim, 2 * channels)
        self.context_bias = nn.Linear(cfg.d_model, channels)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 3, padding=1)
                for _ in range(cfg.grid_layers)
            ]
        )
        self.films = nn.ModuleList(
            [
                nn.Linear(cfg.latent_dim, 2 * channels)
                for _ in range(cfg.grid_layers)
            ]
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        u0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = latent.size(0)
        grid = F.interpolate(
            self.seed.expand(batch, -1, -1, -1),
            size=(self.cfg.nx, self.cfg.nt),
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = self.seed_film(latent).chunk(2, dim=-1)
        grid = grid * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        bias = self.context_bias(context.mean(dim=1))
        grid = grid + bias[:, :, None, None]
        for conv, film in zip(self.convs, self.films):
            gamma, beta = film(latent).chunk(2, dim=-1)
            modulated = (
                grid * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            )
            update = F.silu(conv(modulated))
            grid = grid + update if self.residual else update
        return self.head(grid).squeeze(1)


class SpectralConv2d(nn.Module):
    """2D spectral convolution on the (x, t) grid with low-mode truncation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_t: int,
    ) -> None:
        super().__init__()
        self.modes_x = modes_x
        self.modes_t = modes_t
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_t,
                dtype=torch.cfloat,
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, nx, nt = features.shape
        spectrum = torch.fft.rfft2(features)
        kept_x = min(self.modes_x, nx // 2)
        kept_t = min(self.modes_t, spectrum.size(-1))
        out_spectrum = torch.zeros(
            batch,
            self.weight.size(1),
            spectrum.size(-2),
            spectrum.size(-1),
            dtype=torch.cfloat,
            device=features.device,
        )
        out_spectrum[:, :, :kept_x, :kept_t] = torch.einsum(
            "bixy,ioxy->boxy",
            spectrum[:, :, :kept_x, :kept_t],
            self.weight[:, :, :kept_x, :kept_t],
        )
        return torch.fft.irfft2(out_spectrum, s=(nx, nt))


class FNOConditionedDecoder(nn.Module):
    """FNO-style field decoder conditioned on the manifold latent.

    Scheme 2 (2026-10-25): lift u0 -> broadcast over t -> stacked 2-D spectral
    conv (FNO core) + 1x1 local conv with FiLM(z(0)) -> full field u(x, t).
    The latent ODE is bypassed; time evolution comes from the spectral layers.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = cfg.grid_channels
        self.lift = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.context_bias = nn.Linear(cfg.d_model, channels)
        self.films = nn.ModuleList(
            [
                nn.Linear(cfg.latent_dim, 2 * channels)
                for _ in range(cfg.grid_layers)
            ]
        )
        self.spectral = nn.ModuleList(
            [
                SpectralConv2d(
                    channels,
                    channels,
                    cfg.spectral_modes_x,
                    cfg.spectral_modes_t,
                )
                for _ in range(cfg.grid_layers)
            ]
        )
        self.local = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 1)
                for _ in range(cfg.grid_layers)
            ]
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        u0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if u0 is None:
            raise ValueError("fno_grid decoder requires the initial field u0")
        batch = latent.size(0)
        grid = self.lift(u0.unsqueeze(1))
        grid = grid.unsqueeze(-1).expand(-1, -1, -1, self.cfg.nt)
        bias = self.context_bias(context.mean(dim=1))
        grid = grid + bias[:, :, None, None]
        for spectral, local, film in zip(self.spectral, self.local, self.films):
            gamma, beta = film(latent).chunk(2, dim=-1)
            modulated = (
                grid * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            )
            grid = grid + F.silu(spectral(modulated) + local(modulated))
        return self.head(grid).squeeze(1)


class SpectralGridDecoder(nn.Module):
    """FiLM-conditioned spectral field decoder.

    Each layer pairs a spectral convolution (global, FNO-style) with a 1x1
    local convolution and a residual update, conditioned on the latent state.
    This is the design that takes FNO's representation and adds the manifold /
    attention trunk that a plain FNO does not have.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = cfg.grid_channels
        self.seed = nn.Parameter(
            0.1 * torch.randn(1, channels, cfg.grid_seed, cfg.grid_seed)
        )
        self.seed_film = nn.Linear(cfg.latent_dim, 2 * channels)
        self.context_bias = nn.Linear(cfg.d_model, channels)
        self.spectral = nn.ModuleList(
            [
                SpectralConv2d(
                    channels,
                    channels,
                    cfg.spectral_modes_x,
                    cfg.spectral_modes_t,
                )
                for _ in range(cfg.grid_layers)
            ]
        )
        self.local = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1) for _ in range(cfg.grid_layers)]
        )
        self.films = nn.ModuleList(
            [
                nn.Linear(cfg.latent_dim, 2 * channels)
                for _ in range(cfg.grid_layers)
            ]
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        u0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = latent.size(0)
        grid = F.interpolate(
            self.seed.expand(batch, -1, -1, -1),
            size=(self.cfg.nx, self.cfg.nt),
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = self.seed_film(latent).chunk(2, dim=-1)
        grid = grid * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        bias = self.context_bias(context.mean(dim=1))
        grid = grid + bias[:, :, None, None]
        for spectral, local, film in zip(
            self.spectral,
            self.local,
            self.films,
        ):
            gamma, beta = film(latent).chunk(2, dim=-1)
            modulated = (
                grid * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            )
            grid = grid + F.silu(spectral(modulated) + local(modulated))
        return self.head(grid).squeeze(1)


def build_grid_decoder(cfg: DatasetConfig):
    if cfg.decoder == "grid":
        return GridFieldDecoder(cfg)
    if cfg.decoder == "resnet_grid":
        return GridFieldDecoder(cfg, residual=True)
    if cfg.decoder == "spectral_grid":
        return SpectralGridDecoder(cfg)
    if cfg.decoder == "fno_grid":
        return FNOConditionedDecoder(cfg)
    return None


class BurgersHybridPINN(nn.Module):
    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fourier = Fourier2D(
            bands=cfg.fourier_bands,
            time_bands=cfg.time_fourier_bands,
        )
        self.cnn_only = cfg.arch == "cnn_only"
        self.manifold = None if self.cnn_only else build_manifold(cfg)
        self.patch_tokenizer = cfg.tokenizer == "patch"
        self.patch_len = max(1, cfg.nx // max(cfg.token_count, 1))
        if self.patch_tokenizer:
            self.cnn = None
            self.patch_embed = nn.Linear(self.patch_len, cfg.d_model)
            self.local_feature_channels = cfg.d_model
        elif cfg.cnn_res_blocks > 0:
            # --cnn-res-blocks: conv stem -> residual blocks -> 1x1 mixer.
            width = max(1, cfg.cnn_width)
            self.cnn = nn.Sequential(
                nn.Conv1d(1, width, kernel_size=5, padding=2),
                nn.SiLU(),
                *[
                    ConvResBlock(width, kernel_size=5, dropout=cfg.dropout)
                    for _ in range(cfg.cnn_res_blocks)
                ],
                nn.Conv1d(width, cfg.cnn_channels, kernel_size=1),
                nn.SiLU(),
            )
            self.patch_embed = None
            self.local_feature_channels = cfg.cnn_channels
        else:
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(16, cfg.cnn_channels, kernel_size=5, padding=2),
                nn.SiLU(),
            )
            self.patch_embed = None
            self.local_feature_channels = cfg.cnn_channels
        self.token_proj = nn.Linear(cfg.cnn_channels, cfg.d_model)
        self.token_pos = nn.Parameter(
            torch.zeros(1, cfg.token_count, cfg.d_model)
        )
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        if cfg.use_manifold_cnn:
            self.manifold_token_proj = nn.Linear(
                cfg.latent_dim,
                cfg.token_count * cfg.cnn_channels,
            )
            if cfg.manifold_cnn_res_blocks > 0:
                width = max(1, cfg.manifold_cnn_width or cfg.cnn_channels)
                self.manifold_cnn = nn.Sequential(
                    nn.Conv1d(cfg.cnn_channels, width, kernel_size=3,
                              padding=1),
                    nn.SiLU(),
                    *[
                        ConvResBlock(width, kernel_size=3, dropout=cfg.dropout)
                        for _ in range(cfg.manifold_cnn_res_blocks)
                    ],
                    nn.Conv1d(width, cfg.cnn_channels, kernel_size=1),
                )
            else:
                self.manifold_cnn = nn.Sequential(
                    nn.Conv1d(
                        cfg.cnn_channels,
                        cfg.cnn_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        cfg.cnn_channels,
                        cfg.cnn_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.SiLU(),
                )
            self.manifold_cnn_proj = nn.Linear(
                cfg.cnn_channels,
                cfg.d_model,
            )
        self.context_encoder = (
            None
            if self.cnn_only
            else TransformerEncoder(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
            )
        )
        self.query_proj = nn.Sequential(
            nn.Linear(
                self.fourier.out_dim
                + (0 if self.cnn_only else cfg.latent_dim)
                + 2 * cfg.initial_patch_radius
                + 1
                + (
                    self.local_feature_channels
                    * (2 * cfg.query_local_window + 1)
                    if cfg.query_local_features
                    else 0
                ),
                cfg.d_model,
            ),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.cross_attention = (
            None
            if (self.cnn_only or not cfg.use_cross_attention)
            else ManualAttention(cfg.d_model, cfg.n_heads)
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_width),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_width, 1),
        )
        self.grid_decoder = build_grid_decoder(cfg)
        self.time_fourier = TimeFourier(bands=cfg.time_fourier_bands)
        self.time_context_film = None
        if cfg.time_conditioned_context:
            film = nn.Sequential(
                nn.Linear(self.time_fourier.out_dim, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, 2 * cfg.d_model),
            )
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)
            self.time_context_film = film
        self.vector_field = None
        if cfg.latent_ode:
            if cfg.temporal == "gru":
                self.vector_field = GatedRecurrentTemporal(
                    cfg.latent_dim,
                    cfg.ode_hidden,
                )
            elif cfg.temporal == "ssm":
                self.vector_field = SelectiveStateSpace(
                    cfg.latent_dim,
                    cfg.ode_hidden,
                )
            else:
                self.vector_field = LatentVectorField(
                    cfg.latent_dim,
                    cfg.ode_hidden,
                )
        self.eps_head = None
        if (
            cfg.denoise_sigma > 0.0
            or cfg.denoise_weight > 0.0
            or cfg.embed_flow
        ):
            self.eps_head = nn.Sequential(
                nn.Linear(cfg.latent_dim + 1, 2 * cfg.latent_dim),
                nn.SiLU(),
                nn.Linear(2 * cfg.latent_dim, cfg.latent_dim),
            )
 
    def corrupt_latent(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor | float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.eps_head is None:
            zero = torch.zeros((), device=latent.device)
            return latent, zero, zero
        unit = self.noise_scale()
        if sigma is None:
            if self.cfg.embed_flow:
                sigma = unit * float(
                    torch.rand((), device=latent.device)
                )
            else:
                sigma = self.cfg.denoise_sigma
        if sigma <= 0.0:
            zero = torch.zeros((), device=latent.device)
            return latent, zero, zero
        scale = latent.detach().pow(2).mean().sqrt().clamp_min(1.0e-6)
        noise = sigma * scale * torch.randn_like(latent)
        corrupted = latent + noise
        level = torch.full(
            (latent.size(0), 1),
            float(sigma) / max(unit, 1.0e-6),
            device=latent.device,
        )
        return (
            corrupted,
            noise,
            self.eps_head(torch.cat([corrupted, level], dim=-1)),
        )

    def noise_scale(self) -> float:
        if self.cfg.embed_flow:
            return float(self.cfg.embed_flow_max_sigma)
        return float(self.cfg.denoise_sigma)

    def denoise_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.eps_head is None or self.cfg.denoise_steps <= 0:
            return latent
        scale = latent.detach().pow(2).mean().sqrt().clamp_min(1.0e-6)
        state = latent
        steps = self.cfg.denoise_steps
        for index in range(steps):
            level = 1.0 if not self.cfg.embed_flow else 1.0 - index / steps
            predictor = self.eps_head(
                torch.cat(
                    [
                        state,
                        torch.full(
                            (state.size(0), 1),
                            level,
                            device=state.device,
                        ),
                    ],
                    dim=-1,
                )
            )
            state = state - (
                self.cfg.denoise_rate * self.noise_scale() * scale * predictor
            )
        return state

    def prepare_eval_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.cfg.denoise_eval_sigma > 0.0 and self.eps_head is not None:
            scale = latent.detach().pow(2).mean().sqrt().clamp_min(1.0e-6)
            latent = latent + (
                self.cfg.denoise_eval_sigma * scale * torch.randn_like(latent)
            )
        return self.denoise_latent(latent)

    def integrate_latent(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        steps = max(1, self.cfg.ode_steps)
        step_size = t / steps
        state = latent
        if self.cfg.temporal == "gru":
            for index in range(steps):
                state = self.vector_field(
                    state,
                    time_phase((index + 1) / steps, state),
                )
        elif self.cfg.temporal == "ssm":
            for index in range(steps):
                state = self.vector_field(
                    state,
                    time_phase((index + 1) / steps, state),
                )
        else:
            for index in range(steps):
                state = state + step_size * self.vector_field(state)
        return state

    def encode(self, u0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_with_latent(u0)

    def context_tokens(self, u0: torch.Tensor) -> torch.Tensor:
        if self.patch_tokenizer:
            patches = u0.unfold(1, self.patch_len, self.patch_len)
            return self.patch_embed(patches)
        local = self.cnn(u0.unsqueeze(1))
        local = F.adaptive_avg_pool1d(local, self.cfg.token_count)
        return self.token_proj(local.transpose(1, 2))

    def encode_with_latent(
        self,
        u0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cnn_only:
            latent = torch.zeros(
                u0.size(0),
                self.cfg.latent_dim,
                device=u0.device,
                dtype=u0.dtype,
            )
            return latent, self.local_feature_map(u0)
        latent = self.manifold.encode(u0)
        tokens = self.context_tokens(u0)
        tokens = (
            tokens
            + self.token_pos[:, : tokens.size(1), :]
            + self.latent_proj(latent).unsqueeze(1)
        )
        if self.cfg.use_manifold_cnn:
            manifold_tokens = self.manifold_token_proj(latent)
            manifold_tokens = manifold_tokens.view(
                latent.size(0),
                self.cfg.token_count,
                self.cfg.cnn_channels,
            ).transpose(1, 2)
            manifold_tokens = self.manifold_cnn(manifold_tokens)
            manifold_tokens = self.manifold_cnn_proj(
                manifold_tokens.transpose(1, 2)
            )
            tokens = tokens + manifold_tokens
        return latent, self.context_encoder(tokens)

    def decode_grid(
        self,
        u0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, context = self.encode(u0)
        return latent, context, self.grid_decoder(latent, context, u0)

    def local_feature_map(self, u0: torch.Tensor) -> torch.Tensor:
        if self.patch_tokenizer:
            patches = u0.unfold(1, self.patch_len, self.patch_len)
            tokens = self.patch_embed(patches)
            return F.interpolate(
                tokens.transpose(1, 2),
                size=self.cfg.nx,
                mode="linear",
                align_corners=False,
            )
        return self.cnn(u0.unsqueeze(1))

    def decode_queries(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        initial_value: torch.Tensor,
    ) -> torch.Tensor:
        if latent.size(0) != x.size(0):
            latent = latent.expand(x.size(0), -1)
        if self.vector_field is not None:
            latent = self.integrate_latent(latent, t)
        if self.cnn_only:
            query = torch.cat(
                [self.fourier(x, t), initial_value],
                dim=-1,
            )
        else:
            query = torch.cat(
                [self.fourier(x, t), latent, initial_value],
                dim=-1,
            )
        query = self.query_proj(query).unsqueeze(1)
        if self.time_context_film is not None:
            modulation = self.time_context_film(self.time_fourier(t))
            gamma, beta = modulation.chunk(2, dim=-1)
            context = (
                context * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            )
        if self.cross_attention is None:
            attended = query
        else:
            attended = self.cross_attention(query, context, context)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))
        return self.head(query.squeeze(1))


def load_split(
    path: Path,
    max_samples: int,
) -> torch.Tensor:
    parquet = pq.ParquetFile(path)
    samples: List[np.ndarray] = []
    remaining = max_samples
    for batch in parquet.iter_batches(batch_size=32, columns=["tensor"]):
        column = batch.column(0)
        for index in range(batch.num_rows):
            if remaining <= 0:
                break
            samples.append(
                np.asarray(column[index].as_py(), dtype=np.float32)
            )
            remaining -= 1
        if remaining <= 0:
            break
    if not samples:
        raise RuntimeError(f"No samples loaded from {path}")
    return torch.from_numpy(np.stack(samples, axis=0))


def pretrain_manifold(
    model: BurgersHybridPINN,
    cfg: DatasetConfig,
    train_fields: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(
        model.manifold.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    history: List[Dict[str, float]] = []
    max_grad_norm = 0.0
    skipped_steps = 0
    for step in range(cfg.manifold_steps):
        indices = torch.randint(
            0,
            train_fields.size(0),
            (cfg.batch_size,),
            device=device,
        )
        u0 = train_fields[indices, :, 0, 0]
        latent = model.manifold.encode(u0)
        reconstruction = model.manifold.decode(latent)
        loss = F.mse_loss(reconstruction, u0)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_gradients(model.manifold, cfg.grad_clip_norm)
        max_grad_norm = max(max_grad_norm, grad_norm)
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            continue
        optimizer.step()
        if step % cfg.log_every == 0 or step == cfg.manifold_steps - 1:
            history.append(
                {
                    "step": float(step),
                    "manifold_loss": float(loss.detach()),
                    "grad_norm": grad_norm,
                }
            )
    if history:
        history[-1]["max_grad_norm"] = max_grad_norm
        history[-1]["skipped_steps"] = float(skipped_steps)
    return history


def sample_data_batch(
    fields: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    points_per_sample: int = 1,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    indices = torch.randint(0, fields.size(0), (cfg.batch_size,), device=device)
    point_count = cfg.batch_size * points_per_sample
    x_index = torch.randint(0, cfg.nx, (point_count,), device=device)
    t_index = torch.randint(0, cfg.nt, (point_count,), device=device)
    field_indices = indices.repeat_interleave(points_per_sample)
    u0 = fields[indices, :, 0, 0]
    target = fields[field_indices, x_index, t_index, 0].unsqueeze(1)
    weight = torch.ones_like(target)
    if cfg.data_gradient_weight > 0.0:
        left_index = (x_index - 1).clamp_min(0)
        right_index = (x_index + 1).clamp_max(cfg.nx - 1)
        slope = torch.abs(
            fields[field_indices, right_index, t_index, 0]
            - fields[field_indices, left_index, t_index, 0]
        )
        slope = slope / slope.mean().clamp_min(1.0e-6)
        weight = 1.0 + cfg.data_gradient_weight * slope.clamp_max(5.0)
        weight = weight.unsqueeze(1)
    x = x_index.float() / (cfg.nx - 1)
    t = t_index.float() / (cfg.nt - 1)
    return u0, x.unsqueeze(1), t.unsqueeze(1), target, weight


def sample_physics_batch(
    fields: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    points_per_sample: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, fields.size(0), (cfg.batch_size,), device=device)
    u0 = fields[indices, :, 0, 0]
    if cfg.physics_slope_bias > 0.0:
        slope = torch.abs(u0[:, 1:] - u0[:, :-1])
        slope = slope / slope.mean(dim=1, keepdim=True).clamp_min(1.0e-6)
        weights = torch.ones_like(u0)
        weights[:, :-1] += cfg.physics_slope_bias * slope.clamp_max(5.0)
        x_index = torch.multinomial(
            weights,
            points_per_sample,
            replacement=True,
        ).reshape(-1)
    else:
        x_index = torch.randint(
            0,
            cfg.nx,
            (cfg.batch_size * points_per_sample,),
            device=device,
        )
    t_index = torch.randint(
        0,
        cfg.nt,
        (cfg.batch_size * points_per_sample,),
        device=device,
    )
    x = x_index.float() / (cfg.nx - 1)
    t = t_index.float() / (cfg.nt - 1)
    return u0, x.unsqueeze(1), t.unsqueeze(1)


def pde_residual(
    model: BurgersHybridPINN,
    context: torch.Tensor,
    latent: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    initial_field: torch.Tensor,
    initial_patch_radius: int,
    nu: float,
    create_graph: bool = True,
    nonlinear_scale: float = 1.0,
) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    initial_value = query_local_features(
        model,
        initial_field,
        x,
        initial_patch_radius,
    )
    prediction = model.decode_queries(
        context,
        latent,
        x,
        t,
        initial_value,
    )
    # Three walks over one shared tape.  The t- and x-derivatives must keep
    # their own graph (``create_graph=True, retain_graph=True``) so the second
    # x-derivative can differentiate ``u_x`` and so the training arm can
    # backpropagate *through* the residual.  When the caller only wants the
    # value (``create_graph=False``, i.e. the physics term is switched off) the
    # result is detached, so nothing of this local tape reaches the loss and it
    # is released as soon as the function returns.
    prediction_sum = prediction.sum()
    u_t = torch.autograd.grad(
        prediction_sum,
        t,
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = torch.autograd.grad(
        prediction_sum,
        x,
        create_graph=True,
        retain_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x.sum(),
        x,
        create_graph=create_graph,
    )[0]
    residual = u_t + nonlinear_scale * prediction * u_x - nu * u_xx
    return residual if create_graph else residual.detach()


def periodic_derivative_loss(
    model: BurgersHybridPINN,
    context: torch.Tensor,
    latent: torch.Tensor,
    initial_field: torch.Tensor,
    t: torch.Tensor,
    initial_patch_radius: int,
) -> torch.Tensor:
    left_x = torch.zeros_like(t).requires_grad_(True)
    right_x = torch.ones_like(t).requires_grad_(True)
    left_prediction = model.decode_queries(
        context,
        latent,
        left_x,
        t,
        query_local_features(
            model,
            initial_field,
            left_x,
            initial_patch_radius,
        ),
    )
    right_prediction = model.decode_queries(
        context,
        latent,
        right_x,
        t,
        query_local_features(
            model,
            initial_field,
            right_x,
            initial_patch_radius,
        ),
    )
    left_slope = torch.autograd.grad(
        left_prediction.sum(),
        left_x,
        create_graph=True,
    )[0]
    right_slope = torch.autograd.grad(
        right_prediction.sum(),
        right_x,
        create_graph=True,
    )[0]
    return F.mse_loss(left_slope, right_slope)


def train(
    model: BurgersHybridPINN,
    cfg: DatasetConfig,
    train_fields: torch.Tensor,
    dev_fields: torch.Tensor,
    device: torch.device,
) -> Tuple[
    List[Dict[str, float]],
    Dict[str, torch.Tensor],
    Dict[str, object],
]:
    if cfg.decoder in ("grid", "resnet_grid", "spectral_grid", "fno_grid"):
        return train_grid(model, cfg, train_fields, dev_fields, device)
    if cfg.arch == "rollout":
        return train_rollout(model, cfg, train_fields, dev_fields, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, cfg)
    loss_balancer = LossScaleBalancer(
        enabled=cfg.loss_balance,
        floor=cfg.loss_floor,
    )
    history: List[Dict[str, float]] = []
    best_dev_score = float("inf")
    best_dev_step = -1
    stale_evaluations = 0
    stopped_early = False
    completed_steps = 0
    best_state = copy.deepcopy(model.state_dict())
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    skipped_steps = 0
    clipped_steps = 0

    for step in range(cfg.steps):
        loss_balancer.set_step(step)
        denoise_active = cfg.denoise_sigma > 0.0 and (
            cfg.denoise_until < 0 or step < cfg.denoise_until
        )
        if cfg.embed_flow:
            denoise_active = cfg.denoise_until < 0 or step < cfg.denoise_until
        denoise_sigma = (
            None
            if (denoise_active and cfg.embed_flow)
            else (cfg.denoise_sigma if denoise_active else 0.0)
        )
        data_points = cfg.data_points_per_sample
        u0, x, t, target, data_weight = sample_data_batch(
            train_fields,
            cfg,
            device,
            points_per_sample=data_points,
        )
        latent, context = model.encode(u0)
        clean_latent = latent
        latent, denoise_noise, denoise_prediction = model.corrupt_latent(
            latent,
            denoise_sigma,
        )
        data_latent = latent.repeat_interleave(data_points, dim=0)
        data_context = context.repeat_interleave(data_points, dim=0)
        data_initial_value = query_local_features(
            model,
            u0.repeat_interleave(data_points, dim=0),
            x,
            cfg.initial_patch_radius,
        )
        prediction = model.decode_queries(
            data_context,
            data_latent,
            x,
            t,
            data_initial_value,
        )
        data_loss = torch.sum(
            data_weight * (prediction - target) ** 2
        ) / data_weight.sum()

        # FNO-teacher distillation (dense pseudo-labels on extra query points).
        teacher_fields = getattr(cfg, "_distill_fields", None)
        distill_points = cfg.distill_points_per_sample
        if (
            teacher_fields is not None
            and cfg.distill_weight > 0.0
            and distill_points > 0
        ):
            distill_indices = torch.randint(
                0,
                teacher_fields.size(0),
                (cfg.batch_size,),
                device=device,
            )
            distill_u0 = train_fields[distill_indices, :, 0, 0]
            distill_x_index = torch.randint(
                0,
                cfg.nx,
                (cfg.batch_size * distill_points,),
                device=device,
            )
            distill_t_index = torch.randint(
                0,
                cfg.nt,
                (cfg.batch_size * distill_points,),
                device=device,
            )
            flat_distill = distill_indices.repeat_interleave(distill_points)
            distill_target = teacher_fields[
                flat_distill, distill_x_index, distill_t_index
            ].unsqueeze(1)
            distill_latent, distill_context = model.encode(distill_u0)
            distill_latent, _, _ = model.corrupt_latent(
                distill_latent,
                denoise_sigma,
            )
            distill_x = distill_x_index.float().unsqueeze(1) / (cfg.nx - 1)
            distill_t = distill_t_index.float().unsqueeze(1) / (cfg.nt - 1)
            distill_prediction = model.decode_queries(
                distill_context.repeat_interleave(distill_points, dim=0),
                distill_latent.repeat_interleave(distill_points, dim=0),
                distill_x,
                distill_t,
                query_local_features(
                    model,
                    distill_u0.repeat_interleave(distill_points, dim=0),
                    distill_x,
                    cfg.initial_patch_radius,
                ),
            )
            distill_loss = F.mse_loss(distill_prediction, distill_target)
        else:
            distill_loss = torch.zeros((), device=device)

        physics_points = cfg.physics_points_per_sample
        physics_u0, physics_x, physics_t = sample_physics_batch(
            train_fields,
            cfg,
            device,
            points_per_sample=physics_points,
        )
        physics_latent, physics_context = model.encode(physics_u0)
        physics_latent, _, _ = model.corrupt_latent(
            physics_latent,
            denoise_sigma,
        )
        physics_initial_field = physics_u0.repeat_interleave(
            physics_points,
            dim=0,
        )
        physics_latent = physics_latent.repeat_interleave(
            physics_points,
            dim=0,
        )
        physics_context = physics_context.repeat_interleave(
            physics_points,
            dim=0,
        )
        residual = pde_residual(
            model,
            physics_context,
            physics_latent,
            physics_x,
            physics_t,
            physics_initial_field,
            cfg.initial_patch_radius,
            cfg.nu,
            create_graph=cfg.physics_weight > 0.0,
            nonlinear_scale=cfg.physics_nonlinear_scale,
        )
        physics_loss = F.mse_loss(residual, torch.zeros_like(residual))

        initial_points = cfg.initial_points_per_sample
        initial_point_count = cfg.batch_size * initial_points
        x_index = torch.randint(
            0,
            cfg.nx,
            (initial_point_count,),
            device=device,
        )
        initial_u0 = train_fields[indices := torch.randint(
            0,
            train_fields.size(0),
            (cfg.batch_size,),
            device=device,
        ), :, 0, 0]
        initial_latent, initial_context = model.encode(initial_u0)
        initial_latent, _, _ = model.corrupt_latent(
            initial_latent,
            denoise_sigma,
        )
        initial_x = x_index.float().unsqueeze(1) / (cfg.nx - 1)
        initial_t = torch.zeros_like(initial_x)
        initial_prediction = model.decode_queries(
            initial_context.repeat_interleave(initial_points, dim=0),
            initial_latent.repeat_interleave(initial_points, dim=0),
            initial_x,
            initial_t,
            query_local_features(
                model,
                initial_u0.repeat_interleave(initial_points, dim=0),
                initial_x,
                cfg.initial_patch_radius,
            ),
        )
        initial_target = initial_u0.repeat_interleave(
            initial_points,
            dim=0,
        ).gather(1, x_index.unsqueeze(1))
        initial_loss = F.mse_loss(initial_prediction, initial_target)

        boundary_points = cfg.boundary_points_per_sample
        boundary_t = torch.rand(
            cfg.batch_size * boundary_points,
            1,
            device=device,
        )
        boundary_latent = latent.repeat_interleave(boundary_points, dim=0)
        boundary_context = context.repeat_interleave(
            boundary_points,
            dim=0,
        )
        boundary_left = model.decode_queries(
            boundary_context,
            boundary_latent,
            torch.zeros_like(boundary_t),
            boundary_t,
            query_local_features(
                model,
                u0.repeat_interleave(boundary_points, dim=0),
                torch.zeros_like(boundary_t),
                cfg.initial_patch_radius,
            ),
        )
        boundary_right = model.decode_queries(
            boundary_context,
            boundary_latent,
            torch.ones_like(boundary_t),
            boundary_t,
            query_local_features(
                model,
                u0.repeat_interleave(boundary_points, dim=0),
                torch.ones_like(boundary_t),
                cfg.initial_patch_radius,
            ),
        )
        boundary_value_loss = F.mse_loss(
            boundary_left,
            boundary_right,
        )
        if cfg.periodic_derivative_weight > 0.0:
            boundary_derivative = periodic_derivative_loss(
                model,
                boundary_context,
                boundary_latent,
                u0.repeat_interleave(boundary_points, dim=0),
                boundary_t,
                cfg.initial_patch_radius,
            )
            boundary_loss = (
                boundary_value_loss
                + cfg.periodic_derivative_weight * boundary_derivative
            )
        else:
            boundary_derivative = torch.zeros((), device=device)
            boundary_loss = boundary_value_loss
        if model.manifold is None:
            reconstruction_loss = torch.zeros((), device=device)
        else:
            reconstruction_loss = F.mse_loss(
                model.manifold.decode(clean_latent),
                u0,
            )

        physics_ramp = min(
            1.0,
            max(0.0, (step - cfg.warmup_steps + 1) / max(cfg.warmup_steps, 1)),
        )
        physics_weight = cfg.physics_weight * physics_ramp
        data_term = loss_balancer.scale("data", data_loss)
        physics_term = loss_balancer.scale("physics", physics_loss)
        initial_term = loss_balancer.scale("initial", initial_loss)
        boundary_term = loss_balancer.scale("boundary", boundary_loss)
        reconstruction_term = loss_balancer.scale(
            "reconstruction",
            reconstruction_loss,
        )
        denoise_loss = (
            F.mse_loss(denoise_prediction, denoise_noise)
            if torch.is_tensor(denoise_prediction)
            else torch.zeros((), device=device)
        )
        denoise_term = (
            cfg.denoise_weight
            * loss_balancer.scale("denoise", denoise_loss)
            if denoise_active
            else torch.zeros((), device=device)
        )
        distill_term = cfg.distill_weight * distill_loss
        loss = (
            data_term
            + physics_weight * physics_term
            + 0.1 * initial_term
            + 0.01 * boundary_term
            + 0.001 * reconstruction_term
            + denoise_term
            + distill_term
        )
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped_steps += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_gradients(model, cfg.grad_clip_norm)
        grad_norm_sum += grad_norm
        grad_norm_count += 1
        max_grad_norm = max(max_grad_norm, grad_norm)
        if grad_norm > cfg.grad_clip_norm:
            clipped_steps += 1
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped_steps += 1
            continue
        optimizer.step()
        scheduler.step()

        if (
            step % cfg.eval_every == 0
            or step == cfg.steps - 1
        ):
            dev_metrics = evaluate_split(
                model,
                dev_fields,
                min(cfg.eval_samples, dev_fields.size(0)),
                cfg,
                device,
            )
            if device.type == "cuda":
                # Hand the evaluation's cached blocks back to the driver;
                # resident blocks raise host commit charge on Windows.
                torch.cuda.empty_cache()
            dev_score = (
                dev_metrics["relative_l2"]
                + cfg.validation_physics_weight * dev_metrics["pde_rms"]
            )
            if dev_score < best_dev_score - cfg.early_stop_min_delta:
                best_dev_score = dev_score
                best_dev_step = step
                best_state = copy.deepcopy(model.state_dict())
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            history.append(
                {
                    "step": float(step),
                    "total": float(loss.detach()),
                    "data": float(data_loss.detach()),
                    "physics": float(physics_loss.detach()),
                    "initial": float(initial_loss.detach()),
                    "boundary": float(boundary_loss.detach()),
                    "boundary_value": float(
                        boundary_value_loss.detach()
                    ),
                    "boundary_derivative": float(
                        boundary_derivative.detach()
                    ),
                    "reconstruction": float(reconstruction_loss.detach()),
                    "denoise": float(denoise_loss.detach()),
                    "distill": float(distill_loss.detach()),
                    "data_scaled": float(data_term.detach()),
                    "physics_scaled": float(physics_term.detach()),
                    "initial_scaled": float(initial_term.detach()),
                    "boundary_scaled": float(boundary_term.detach()),
                    "reconstruction_scaled": float(
                        reconstruction_term.detach()
                    ),
                    "physics_ramp": physics_ramp,
                    "dev_relative_l2": dev_metrics["relative_l2"],
                    "dev_pde_rms": dev_metrics["pde_rms"],
                    "dev_score": dev_score,
                    "grad_norm": grad_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            completed_steps = step + 1
            if (
                cfg.early_stop_patience > 0
                and stale_evaluations >= cfg.early_stop_patience
            ):
                stopped_early = True
                break
    if completed_steps == 0:
        completed_steps = cfg.steps
    model.load_state_dict(best_state)
    stability = {
        "max_grad_norm": max_grad_norm,
        "mean_grad_norm": (
            grad_norm_sum / grad_norm_count
            if grad_norm_count > 0
            else 0.0
        ),
        "clipped_steps": float(clipped_steps),
        "skipped_steps": float(skipped_steps),
        "grad_clip_norm": cfg.grad_clip_norm,
        "loss_balance_enabled": cfg.loss_balance,
        "loss_references": loss_balancer.references,
        "best_dev_score": best_dev_score,
        "best_dev_step": float(best_dev_step),
        "stopped_early": stopped_early,
        "completed_steps": float(completed_steps),
        "stale_evaluations": float(stale_evaluations),
    }
    return history, best_state, stability


def evaluate_split(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    n_samples: int,
    cfg: DatasetConfig,
    device: torch.device,
) -> Dict[str, float]:
    if cfg.decoder in ("grid", "resnet_grid", "spectral_grid", "fno_grid"):
        return evaluate_split_grid(model, fields, n_samples, cfg, device)
    if cfg.arch == "rollout":
        return evaluate_rollout(model, fields, n_samples, cfg, device)
    model.eval()
    x_axis = torch.linspace(0.0, 1.0, cfg.nx, device=device)
    t_axis = torch.linspace(0.0, 1.0, cfg.nt, device=device)
    x_flat = x_axis.repeat_interleave(cfg.nt).unsqueeze(1)
    t_flat = t_axis.repeat(cfg.nx).unsqueeze(1)
    point_count = x_flat.size(0)

    relative_errors: List[float] = []
    # Evaluation never needs gradients.  Without this guard the full-grid
    # forward built a graph over nx * nt query rows (~26k) per sample, which
    # alone exceeded the 8 GB card once the tokenizer's feature map was part of
    # it.  The physics diagnostic below still needs autograd, so it is the only
    # part left outside the guard.
    with torch.no_grad():
        for sample_index in range(min(n_samples, fields.size(0))):
            field = fields[sample_index]
            u0 = field[:, 0, 0].unsqueeze(0)
            latent, context = model.encode(u0)
            latent = model.prepare_eval_latent(latent)
            chunks: List[torch.Tensor] = []
            for start in range(0, point_count, 4096):
                stop = min(start + 4096, point_count)
                query_initial = query_local_features(
                    model,
                    u0.expand(stop - start, -1),
                    x_flat[start:stop],
                    cfg.initial_patch_radius,
                )
                chunks.append(
                    model.decode_queries(
                        context,
                        expand_latent(latent, stop - start),
                        x_flat[start:stop],
                        t_flat[start:stop],
                        query_initial,
                    )
                )
            prediction = torch.cat(chunks, dim=0)
            target = field.reshape(-1, 1)
            relative_errors.append(
                float(
                    (
                        torch.linalg.vector_norm(prediction - target)
                        / torch.linalg.vector_norm(target).clamp_min(1.0e-8)
                    )
                )
            )

    pde_count = min(128, fields.size(0))
    physics_u0 = fields[:pde_count, :, 0, 0]
    physics_latent, physics_context = model.encode(physics_u0)
    physics_x = torch.rand(pde_count, 1, device=device)
    physics_t = torch.rand(pde_count, 1, device=device)
    residuals = pde_residual(
        model,
        physics_context,
        physics_latent,
        physics_x,
        physics_t,
        physics_u0,
        cfg.initial_patch_radius,
        cfg.nu,
    )
    pde_rms = float(torch.sqrt(torch.mean(residuals**2)).detach())
    model.train()
    return {
        "relative_l2": float(np.mean(relative_errors)),
        "pde_rms": pde_rms,
    }


class RolloutTemporalCell(nn.Module):
    """Recurrent cell that carries the latent trajectory state forward in time."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.kind = cfg.temporal if cfg.temporal in ("gru", "ssm") else "gru"
        dim = cfg.latent_dim
        hidden = cfg.ode_hidden
        if self.kind == "ssm":
            self.to_gates = nn.Sequential(
                nn.Linear(dim + 3, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * dim),
            )
        else:
            self.cell = nn.GRUCell(dim + 3, dim)

    def forward(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "ssm":
            delta, value = self.to_gates(
                torch.cat([latent, phase], dim=-1)
            ).chunk(2, dim=-1)
            decay = torch.exp(-F.softplus(delta))
            return decay * state + (1.0 - decay) * torch.tanh(value)
        return self.cell(torch.cat([latent, phase], dim=-1), state)


def build_vector_field(cfg: DatasetConfig) -> nn.Module:
    if cfg.temporal == "gru":
        return GatedRecurrentTemporal(cfg.latent_dim, cfg.ode_hidden)
    if cfg.temporal == "ssm":
        return SelectiveStateSpace(cfg.latent_dim, cfg.ode_hidden)
    return LatentVectorField(cfg.latent_dim, cfg.ode_hidden)


class BurgersOperatorPINN(nn.Module):
    """Minimal operator: encoder -> latent flow -> coordinate decoder.

    By default there is no convolution, no attention and no local
    initial-field feature: the whole field is carried by the latent trajectory
    z(t), and the decoder only sees z(t) with the query coordinates.

    With ``cfg.operator_attention`` (``--operator-attention``) a CNN tokenizer,
    a Transformer encoder and a cross-attention query step are added back, so
    the decoder can read a token sequence instead of relying on the latent
    alone. Convolution is kept in this path.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.manifold = build_manifold(cfg)
        self.fourier = Fourier2D(
            bands=cfg.fourier_bands,
            time_bands=cfg.time_fourier_bands,
        )
        self.time_fourier = TimeFourier(bands=cfg.time_fourier_bands)
        self.vector_field = build_vector_field(cfg)
        # Optional attention block: the operator has no CNN tokenizer, so when
        # attention is enabled the token sequence comes from linear patch
        # embeddings instead. Off by default so the plain operator remains the
        # no-attention ablation baseline.
        self.operator_attention = bool(cfg.operator_attention)
        if self.operator_attention:
            # CNN tokenizer stays in place: local feature extraction remains
            # convolutional, the pooled tokens are what attention reads.
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(16, cfg.cnn_channels, kernel_size=5, padding=2),
                nn.SiLU(),
            )
            self.token_proj = nn.Linear(cfg.cnn_channels, cfg.d_model)
            self.token_pos = nn.Parameter(
                torch.zeros(1, cfg.token_count, cfg.d_model)
            )
            self.latent_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
            self.context_encoder = TransformerEncoder(
                cfg.d_model,
                cfg.n_heads,
                cfg.n_layers,
            )
            self.cross_attention = ManualAttention(cfg.d_model, cfg.n_heads)
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, 2 * cfg.d_model),
                nn.SiLU(),
                nn.Linear(2 * cfg.d_model, cfg.d_model),
            )
        else:
            self.context_encoder = None
            self.cross_attention = None
        self.query_proj = nn.Sequential(
            nn.Linear(
                self.fourier.out_dim + cfg.latent_dim,
                cfg.d_model,
            ),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_width),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_width, 1),
        )
        self.eps_head = None
        if (
            cfg.denoise_sigma > 0.0
            or cfg.denoise_weight > 0.0
            or cfg.embed_flow
        ):
            self.eps_head = nn.Sequential(
                nn.Linear(cfg.latent_dim + 1, 2 * cfg.latent_dim),
                nn.SiLU(),
                nn.Linear(2 * cfg.latent_dim, cfg.latent_dim),
            )

    def integrate_latent(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        steps = max(1, self.cfg.ode_steps)
        step_size = t / steps
        state = latent
        if self.cfg.temporal in ("gru", "ssm"):
            for index in range(steps):
                state = self.vector_field(
                    state,
                    time_phase((index + 1) / steps, state),
                )
        else:
            for index in range(steps):
                state = state + step_size * self.vector_field(state)
        return state

    def encode(
        self,
        u0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.manifold.encode(u0)
        if not self.operator_attention:
            # The coordinate decoder ignores the context, but callers expand
            # it, so keep a single dummy element per trajectory.
            context = torch.zeros(
                u0.size(0),
                1,
                1,
                device=u0.device,
                dtype=u0.dtype,
            )
            return latent, context
        # CNN tokenizer: keep local convolutional features, then let the
        # Transformer attend over the pooled spatial tokens.
        local = self.cnn(u0.unsqueeze(1))
        local = F.adaptive_avg_pool1d(local, self.cfg.token_count)
        tokens = self.token_proj(local.transpose(1, 2))
        tokens = (
            tokens
            + self.token_pos[:, : tokens.size(1), :]
            + self.latent_proj(latent).unsqueeze(1)
        )
        return latent, self.context_encoder(tokens)

    def decode_queries(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        initial_value: torch.Tensor,
    ) -> torch.Tensor:
        if latent.size(0) != x.size(0):
            latent = latent.expand(x.size(0), -1)
        latent = self.integrate_latent(latent, t)
        query = self.query_proj(
            torch.cat([self.fourier(x, t), latent], dim=-1)
        )
        if self.cross_attention is None:
            return self.head(query)
        query = query.unsqueeze(1)
        attended = self.cross_attention(query, context, context)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))
        return self.head(query.squeeze(1))

    def noise_scale(self) -> float:
        if self.cfg.embed_flow:
            return float(self.cfg.embed_flow_max_sigma)
        return float(self.cfg.denoise_sigma)

    def corrupt_latent(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor | float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.eps_head is None:
            zero = torch.zeros((), device=latent.device)
            return latent, zero, zero
        unit = self.noise_scale()
        if sigma is None:
            if self.cfg.embed_flow:
                sigma = unit * float(
                    torch.rand((), device=latent.device)
                )
            else:
                sigma = self.cfg.denoise_sigma
        if sigma <= 0.0:
            zero = torch.zeros((), device=latent.device)
            return latent, zero, zero
        scale = latent.detach().pow(2).mean().sqrt().clamp_min(1.0e-6)
        noise = sigma * scale * torch.randn_like(latent)
        corrupted = latent + noise
        level = torch.full(
            (latent.size(0), 1),
            float(sigma) / max(unit, 1.0e-6),
            device=latent.device,
        )
        return (
            corrupted,
            noise,
            self.eps_head(torch.cat([corrupted, level], dim=-1)),
        )

    def denoise_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def prepare_eval_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class BurgersRolloutPINN(BurgersHybridPINN):
    """Field-space time-marching model: predict u(., t+dt) from u(., t)."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__(cfg)
        self.temporal_cell = RolloutTemporalCell(cfg)
        self.register_buffer(
            "x_grid",
            torch.linspace(0.0, 1.0, cfg.nx).view(1, cfg.nx, 1),
        )

    def march(
        self,
        u_start: torch.Tensor,
        steps: int,
        start_index: int = 0,
        truth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        dt = 1.0 / (cfg.nt - 1)
        state = self.manifold.encode(u_start)
        field = u_start
        fields: List[torch.Tensor] = [field]
        for index in range(steps):
            if (
                truth is not None
                and self.training
                and cfg.rollout_teacher_forcing > 0.0
            ):
                mask = torch.rand(
                    field.size(0),
                    1,
                    device=field.device,
                ) < cfg.rollout_teacher_forcing
                field = torch.where(mask, truth[index], field)
            level = min(1.0, (start_index + index + 1) * dt)
            latent = self.manifold.encode(field)
            state = self.temporal_cell(
                latent,
                state,
                time_phase(level, state),
            )
            tokens = self.context_tokens(field)
            context = self.context_encoder(
                tokens + self.token_pos[:, : tokens.size(1), :]
            )
            field = self.decode_field(context, state, field, level)
            fields.append(field)
        return torch.stack(fields, dim=0)

    def decode_field(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        field: torch.Tensor,
        level: float,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch = field.size(0)
        x = (
            self.x_grid.to(field.device)
            .expand(batch, cfg.nx, 1)
            .reshape(-1, 1)
        )
        t = torch.full_like(x, float(level))
        patch = interpolate_initial_patch(
            field.repeat_interleave(cfg.nx, dim=0),
            x,
            cfg.initial_patch_radius,
        )
        if cfg.query_local_features and cfg.query_local_window <= 0:
            # Queries are the fixed x grid, so the feature map can be used
            # directly instead of materialising nx copies of it and
            # interpolating at every grid point.
            sampled = self.local_feature_map(field).transpose(1, 2)
            initial = torch.cat(
                [patch, sampled.reshape(batch * cfg.nx, -1)],
                dim=-1,
            )
        elif cfg.query_local_features:
            initial = query_local_features(
                self,
                field.repeat_interleave(cfg.nx, dim=0),
                x,
                cfg.initial_patch_radius,
            )
        else:
            initial = patch
        prediction = self.decode_queries(
            context.repeat_interleave(cfg.nx, dim=0),
            state.repeat_interleave(cfg.nx, dim=0),
            x,
            t,
            initial,
        )
        prediction = prediction.view(batch, cfg.nx)
        if cfg.rollout_delta:
            return field + prediction
        return prediction


def rollout_residual(
    trajectory: torch.Tensor,
    cfg: DatasetConfig,
    nu: float,
) -> torch.Tensor:
    """Finite-difference Burgers residual along a rolled-out trajectory."""
    dx = 1.0 / (cfg.nx - 1)
    dt = 1.0 / (cfg.nt - 1)
    current = trajectory[1:]
    previous = trajectory[:-1]
    u_t = (current - previous) / dt
    u_x = (current[:, :, 2:] - current[:, :, :-2]) / (2.0 * dx)
    u_xx = (
        current[:, :, 2:] - 2.0 * current[:, :, 1:-1] + current[:, :, :-2]
    ) / (dx * dx)
    interior = current[:, :, 1:-1]
    return u_t[:, :, 1:-1] + interior * u_x - nu * u_xx


def grid_pde_residual(
    prediction: torch.Tensor,
    cfg: DatasetConfig,
    nu: float,
) -> torch.Tensor:
    dx = 1.0 / (cfg.nx - 1)
    dt = 1.0 / (cfg.nt - 1)
    interior = prediction[:, 1:-1, 1:-1]
    u_t = (prediction[:, 1:-1, 2:] - prediction[:, 1:-1, :-2]) / (2.0 * dt)
    u_x = (prediction[:, 2:, 1:-1] - prediction[:, :-2, 1:-1]) / (2.0 * dx)
    u_xx = (
        prediction[:, 2:, 1:-1]
        - 2.0 * interior
        + prediction[:, :-2, 1:-1]
    ) / (dx * dx)
    return u_t + interior * u_x - nu * u_xx


def train_grid(
    model: BurgersHybridPINN,
    cfg: DatasetConfig,
    train_fields: torch.Tensor,
    dev_fields: torch.Tensor,
    device: torch.device,
) -> Tuple[
    List[Dict[str, float]],
    Dict[str, torch.Tensor],
    Dict[str, object],
]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, cfg)
    loss_balancer = LossScaleBalancer(
        enabled=cfg.loss_balance,
        floor=cfg.loss_floor,
    )
    history: List[Dict[str, float]] = []
    best_dev_score = float("inf")
    best_dev_step = -1
    stale_evaluations = 0
    stopped_early = False
    completed_steps = 0
    best_state = copy.deepcopy(model.state_dict())
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    skipped_steps = 0
    clipped_steps = 0

    for step in range(cfg.steps):
        loss_balancer.set_step(step)
        indices = torch.randint(
            0,
            train_fields.size(0),
            (cfg.batch_size,),
            device=device,
        )
        u0 = train_fields[indices, :, 0, 0]
        target = train_fields[indices, :, :, 0]
        latent, _, prediction = model.decode_grid(u0)
        data_loss = F.mse_loss(prediction, target)
        initial_loss = F.mse_loss(prediction[:, :, 0], u0)
        boundary_loss = F.mse_loss(prediction[:, 0, :], prediction[:, -1, :])
        residual = grid_pde_residual(prediction, cfg, cfg.nu)
        physics_loss = F.mse_loss(residual, torch.zeros_like(residual))
        reconstruction_loss = F.mse_loss(model.manifold.decode(latent), u0)

        physics_ramp = min(
            1.0,
            max(0.0, (step - cfg.warmup_steps + 1) / max(cfg.warmup_steps, 1)),
        )
        total = (
            loss_balancer.scale("data", data_loss)
            + cfg.physics_weight
            * physics_ramp
            * loss_balancer.scale("physics", physics_loss)
            + loss_balancer.scale("initial", initial_loss)
            + loss_balancer.scale("boundary", boundary_loss)
            + loss_balancer.scale("reconstruction", reconstruction_loss)
        )

        if not torch.isfinite(total):
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            completed_steps = step + 1
            continue

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = clip_gradients(model, cfg.grad_clip_norm)
        max_grad_norm = max(max_grad_norm, grad_norm)
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            completed_steps = step + 1
            continue
        if grad_norm > cfg.grad_clip_norm:
            clipped_steps += 1
        optimizer.step()
        scheduler.step()
        grad_norm_sum += grad_norm
        grad_norm_count += 1
        completed_steps = step + 1

        if cfg.eval_every > 0 and (
            (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1
        ):
            dev_metrics = evaluate_split_grid(
                model,
                dev_fields,
                min(cfg.eval_samples, dev_fields.size(0)),
                cfg,
                device,
            )
            score = (
                dev_metrics["relative_l2"]
                + cfg.validation_physics_weight * dev_metrics["pde_rms"]
            )
            history.append(
                {
                    "step": float(step + 1),
                    "total": float(total.detach()),
                    "data": float(data_loss.detach()),
                    "physics": float(physics_loss.detach()),
                    "initial": float(initial_loss.detach()),
                    "boundary": float(boundary_loss.detach()),
                    "reconstruction": float(reconstruction_loss.detach()),
                    "dev_relative_l2": dev_metrics["relative_l2"],
                    "dev_pde_rms": dev_metrics["pde_rms"],
                    "grad_norm": grad_norm,
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )
            if score < best_dev_score - cfg.early_stop_min_delta:
                best_dev_score = score
                best_dev_step = step + 1
                stale_evaluations = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale_evaluations += 1
                if (
                    cfg.early_stop_patience > 0
                    and stale_evaluations >= cfg.early_stop_patience
                ):
                    stopped_early = True
                    break

    model.load_state_dict(best_state)
    stability = {
        "max_grad_norm": max_grad_norm,
        "mean_grad_norm": (
            grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
        ),
        "clipped_steps": float(clipped_steps),
        "skipped_steps": float(skipped_steps),
        "loss_balance": loss_balancer.enabled,
        "loss_references": loss_balancer.references,
        "best_dev_score": best_dev_score,
        "best_dev_step": float(best_dev_step),
        "stopped_early": stopped_early,
        "completed_steps": float(completed_steps),
        "stale_evaluations": float(stale_evaluations),
    }
    return history, best_state, stability


def evaluate_split_grid(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    n_samples: int,
    cfg: DatasetConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    count = min(n_samples, fields.size(0))
    relative_errors: List[float] = []
    pde_values: List[float] = []
    for start in range(0, count, cfg.batch_size):
        stop = min(start + cfg.batch_size, count)
        u0 = fields[start:stop, :, 0, 0]
        target = fields[start:stop, :, :, 0]
        with torch.no_grad():
            _, _, prediction = model.decode_grid(u0)
        flat_prediction = prediction.reshape(stop - start, -1)
        flat_target = target.reshape(stop - start, -1)
        per_sample = torch.linalg.vector_norm(
            flat_prediction - flat_target,
            dim=1,
        ) / torch.linalg.vector_norm(flat_target, dim=1).clamp_min(1.0e-8)
        relative_errors.extend(per_sample.tolist())
        residual = grid_pde_residual(prediction, cfg, cfg.nu)
        pde_values.append(float(torch.sqrt(torch.mean(residual**2))))
    model.train()
    return {
        "relative_l2": float(np.mean(relative_errors)),
        "pde_rms": float(np.mean(pde_values)),
    }


def plot_prediction_rollout(
    model: BurgersHybridPINN,
    field: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    roll = model.march(field[:, 0, 0].unsqueeze(0), cfg.nt - 1, 0)
    prediction = roll.permute(1, 2, 0)[0].detach().cpu().numpy()
    x_axis = np.linspace(0.0, 1.0, cfg.nx)
    time_indices = (0, 25, 50, 75, 100)
    figure, axes = plt.subplots(1, len(time_indices), figsize=(14, 3.8))
    for axis, time_index in zip(axes, time_indices):
        target = field[:, time_index, 0].cpu().numpy()
        axis.plot(x_axis, target, color="black", label="dataset")
        axis.plot(x_axis, prediction[:, time_index], linestyle="--", label="rollout")
        axis.set_title(f"t index = {time_index}")
        axis.set_xlabel("x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_rollout(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    n_samples: int,
    cfg: DatasetConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    count = min(n_samples, fields.size(0))
    chunk = max(1, cfg.batch_size // 2)
    relative_errors: List[float] = []
    pde_values: List[float] = []
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        u0 = fields[start:stop, :, 0, 0]
        with torch.no_grad():
            rolled = model.march(u0, cfg.nt - 1, 0)
        prediction = rolled.permute(1, 2, 0)
        target = fields[start:stop, :, :, 0]
        prediction_flat = prediction.reshape(stop - start, -1)
        target_flat = target.reshape(stop - start, -1)
        per_sample = torch.linalg.vector_norm(
            prediction_flat - target_flat,
            dim=1,
        ) / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(1.0e-8)
        relative_errors.extend(per_sample.tolist())
        residual = grid_pde_residual(prediction, cfg, cfg.nu)
        pde_values.append(float(torch.sqrt(torch.mean(residual**2))))
    model.train()
    return {
        "relative_l2": float(np.mean(relative_errors)),
        "pde_rms": float(np.mean(pde_values)),
    }


def train_rollout(
    model: BurgersHybridPINN,
    cfg: DatasetConfig,
    train_fields: torch.Tensor,
    dev_fields: torch.Tensor,
    device: torch.device,
) -> Tuple[
    List[Dict[str, float]],
    Dict[str, torch.Tensor],
    Dict[str, object],
]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, cfg)
    loss_balancer = LossScaleBalancer(
        enabled=cfg.loss_balance,
        floor=cfg.loss_floor,
    )
    history: List[Dict[str, float]] = []
    best_dev_score = float("inf")
    best_dev_step = -1
    stale_evaluations = 0
    stopped_early = False
    completed_steps = 0
    best_state = copy.deepcopy(model.state_dict())
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    skipped_steps = 0
    clipped_steps = 0
    horizon = cfg.nt - 1
    window = max(1, min(cfg.rollout_steps, horizon))
    starts = horizon - window + 1

    for step in range(cfg.steps):
        loss_balancer.set_step(step)
        indices = torch.randint(
            0,
            train_fields.size(0),
            (cfg.batch_size,),
            device=device,
        )
        start_index = int(
            torch.randint(0, starts, (1,), device=device).item()
        )
        u_start = train_fields[indices, :, start_index, 0]
        target = train_fields[
            indices,
            :,
            start_index + 1 : start_index + 1 + window,
            0,
        ].permute(2, 0, 1)
        truth = train_fields[
            indices,
            :,
            start_index : start_index + 1 + window,
            0,
        ].permute(2, 0, 1)
        trajectory = model.march(
            u_start,
            window,
            start_index,
            truth=truth,
        )
        data_loss = F.mse_loss(trajectory[1:], target)
        residual = rollout_residual(trajectory, cfg, cfg.nu)
        physics_loss = F.mse_loss(residual, torch.zeros_like(residual))
        if model.manifold is None:
            reconstruction_loss = torch.zeros((), device=device)
        else:
            reconstruction_loss = F.mse_loss(
                model.manifold.decode(model.manifold.encode(u_start)),
                u_start,
            )

        physics_ramp = min(
            1.0,
            max(0.0, (step - cfg.warmup_steps + 1) / max(cfg.warmup_steps, 1)),
        )
        data_term = loss_balancer.scale("data", data_loss)
        physics_term = loss_balancer.scale("physics", physics_loss)
        reconstruction_term = loss_balancer.scale(
            "reconstruction",
            reconstruction_loss,
        )
        loss = (
            data_term
            + cfg.physics_weight * physics_ramp * physics_term
            + 0.001 * reconstruction_term
        )
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped_steps += 1
            completed_steps = step + 1
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_gradients(model, cfg.grad_clip_norm)
        max_grad_norm = max(max_grad_norm, grad_norm)
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped_steps += 1
            completed_steps = step + 1
            continue
        if grad_norm > cfg.grad_clip_norm:
            clipped_steps += 1
        optimizer.step()
        scheduler.step()
        grad_norm_sum += grad_norm
        grad_norm_count += 1
        completed_steps = step + 1

        if cfg.eval_every > 0 and (
            (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1
        ):
            dev_metrics = evaluate_rollout(
                model,
                dev_fields,
                min(cfg.eval_samples, dev_fields.size(0)),
                cfg,
                device,
            )
            score = (
                dev_metrics["relative_l2"]
                + cfg.validation_physics_weight * dev_metrics["pde_rms"]
            )
            history.append(
                {
                    "step": float(step + 1),
                    "total": float(loss.detach()),
                    "data": float(data_loss.detach()),
                    "physics": float(physics_loss.detach()),
                    "reconstruction": float(reconstruction_loss.detach()),
                    "dev_relative_l2": dev_metrics["relative_l2"],
                    "dev_pde_rms": dev_metrics["pde_rms"],
                    "grad_norm": grad_norm,
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )
            if score < best_dev_score - cfg.early_stop_min_delta:
                best_dev_score = score
                best_dev_step = step + 1
                stale_evaluations = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale_evaluations += 1
                if (
                    cfg.early_stop_patience > 0
                    and stale_evaluations >= cfg.early_stop_patience
                ):
                    stopped_early = True
                    break

    model.load_state_dict(best_state)
    stability = {
        "max_grad_norm": max_grad_norm,
        "mean_grad_norm": (
            grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
        ),
        "clipped_steps": float(clipped_steps),
        "skipped_steps": float(skipped_steps),
        "loss_balance": loss_balancer.enabled,
        "loss_references": loss_balancer.references,
        "best_dev_score": best_dev_score,
        "best_dev_step": float(best_dev_step),
        "stopped_early": stopped_early,
        "completed_steps": float(completed_steps),
        "stale_evaluations": float(stale_evaluations),
    }
    return history, best_state, stability


def plot_prediction_grid(
    model: BurgersHybridPINN,
    field: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    if cfg.decoder in ("grid", "resnet_grid", "spectral_grid", "fno_grid"):
        plot_prediction_grid(model, field, cfg, device, path)
        return
    if cfg.arch == "rollout":
        plot_prediction_rollout(model, field, cfg, device, path)
        return
    model.eval()
    u0 = field[:, 0, 0].unsqueeze(0)
    with torch.no_grad():
        _, _, prediction = model.decode_grid(u0)
    prediction = prediction[0].cpu().numpy()
    x_axis = np.linspace(0.0, 1.0, cfg.nx)
    time_indices = (0, 25, 50, 75, 100)
    figure, axes = plt.subplots(1, len(time_indices), figsize=(14, 3.8))
    for axis, time_index in zip(axes, time_indices):
        target = field[:, time_index, 0].cpu().numpy()
        axis.plot(x_axis, target, color="black", label="dataset")
        axis.plot(
            x_axis,
            prediction[:, time_index],
            linestyle="--",
            label="hybrid",
        )
        axis.set_title(f"t index = {time_index}")
        axis.set_xlabel("x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_training(history: List[Dict[str, float]], path: Path) -> None:
    steps = [row["step"] for row in history]
    total = [row["total"] for row in history]
    dev = [row["dev_relative_l2"] for row in history]
    figure, axis = plt.subplots(1, 2, figsize=(12, 4))
    axis[0].plot(steps, total, label="train total loss")
    axis[0].set_yscale("log")
    axis[0].set_xlabel("step")
    axis[0].set_ylabel("loss")
    axis[0].legend()
    axis[1].plot(steps, dev, label="dev relative L2")
    axis[1].set_yscale("log")
    axis[1].set_xlabel("step")
    axis[1].set_ylabel("relative L2")
    axis[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_prediction(
    model: BurgersHybridPINN,
    field: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    model.eval()
    u0 = field[:, 0, 0].unsqueeze(0)
    latent, context = model.encode(u0)
    latent = model.prepare_eval_latent(latent)
    x_axis = torch.linspace(0.0, 1.0, cfg.nx, device=device)
    time_indices = (0, 25, 50, 75, 100)
    figure, axes = plt.subplots(1, len(time_indices), figsize=(14, 3.8))
    for axis, time_index in zip(axes, time_indices):
        t_value = float(time_index) / (cfg.nt - 1)
        x = x_axis.unsqueeze(1)
        t = torch.full_like(x, t_value)
        with torch.no_grad():
            prediction = model.decode_queries(
                context,
                latent,
                x,
                t,
                query_local_features(
                    model,
                    u0.expand(cfg.nx, -1),
                    x,
                    cfg.initial_patch_radius,
                ),
            )
        target = field[:, time_index, 0].unsqueeze(1)
        axis.plot(
            x_axis.cpu().numpy(),
            target[:, 0].cpu().numpy(),
            color="black",
            label="dataset",
        )
        axis.plot(
            x_axis.cpu().numpy(),
            prediction[:, 0].cpu().numpy(),
            linestyle="--",
            label="hybrid",
        )
        axis.set_title(f"t index = {time_index}")
        axis.set_xlabel("x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def diagnose(train_metric: float, test_metric: float) -> str:
    if train_metric > 0.10 and test_metric > 0.10:
        return "underfit"
    if train_metric < 0.03 and test_metric > 1.5 * train_metric:
        return "overfit"
    return "balanced"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("stable", "balanced", "capacity", "wide"),
        default="balanced",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--manifold-steps", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--dev-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-points", type=int, default=None)
    parser.add_argument("--physics-points", type=int, default=None)
    parser.add_argument("--initial-points", type=int, default=None)
    parser.add_argument("--boundary-points", type=int, default=None)
    parser.add_argument("--initial-patch-radius", type=int, default=None)
    parser.add_argument("--physics-slope-bias", type=float, default=None)
    parser.add_argument("--data-gradient-weight", type=float, default=None)
    parser.add_argument(
        "--periodic-derivative-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--manifold-hidden", type=int, default=None)
    parser.add_argument(
        "--manifold-arch",
        choices=("mlp", "conv"),
        default=None,
    )
    parser.add_argument("--conv-ae-channels", type=int, default=None)
    parser.add_argument("--denoise-sigma", type=float, default=None)
    parser.add_argument("--denoise-weight", type=float, default=None)
    parser.add_argument("--denoise-steps", type=int, default=None)
    parser.add_argument("--denoise-rate", type=float, default=None)
    parser.add_argument("--denoise-eval-sigma", type=float, default=None)
    parser.add_argument("--denoise-until", type=int, default=None)
    parser.add_argument("--embed-flow", action="store_true", default=None)
    parser.add_argument("--embed-flow-max-sigma", type=float, default=None)
    parser.add_argument("--embed-flow-weight", type=float, default=None)
    parser.add_argument(
        "--cross-attention",
        dest="use_cross_attention",
        action="store_true",
    )
    parser.add_argument(
        "--no-cross-attention",
        dest="use_cross_attention",
        action="store_false",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("cnn", "patch"),
        default=None,
    )
    parser.add_argument(
        "--arch",
        choices=("full", "cnn_only", "rollout", "operator", "latent_grid"),
        default=None,
    )
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--latent-levels", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--rollout-teacher-forcing", type=float, default=None)
    parser.add_argument("--rollout-delta", action="store_true", default=None)
    parser.add_argument("--operator-attention", action="store_true", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fourier-bands", type=int, default=None)
    parser.add_argument("--time-fourier-bands", type=int, default=None)
    parser.add_argument("--head-width", type=int, default=None)
    parser.add_argument("--query-local-window", type=int, default=None)
    parser.add_argument(
        "--decoder",
        choices=("query", "grid", "resnet_grid", "spectral_grid", "fno_grid"),
        default=None,
    )
    parser.add_argument("--grid-channels", type=int, default=None)
    parser.add_argument("--grid-layers", type=int, default=None)
    parser.add_argument("--grid-seed", type=int, default=None)
    parser.add_argument("--spectral-modes-x", type=int, default=None)
    parser.add_argument("--spectral-modes-t", type=int, default=None)
    parser.add_argument("--latent-ode", action="store_true", default=None)
    parser.add_argument("--ode-steps", type=int, default=None)
    parser.add_argument("--ode-hidden", type=int, default=None)
    parser.add_argument(
        "--temporal",
        choices=("ode", "gru", "ssm"),
        default=None,
    )
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--cnn-channels", type=int, default=None)
    parser.add_argument("--token-count", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--loss-floor", type=float, default=None)
    parser.add_argument("--loss-balance-warmup", type=int, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None,
                        help="override the PDE residual viscosity")
    parser.add_argument("--physics-nonlinear-scale", type=float,
                        default=None,
                        help="coefficient of u*u_x in the PDE residual"
                             " (default 1.0; PDEBench flux form is 2.0)")
    parser.add_argument(
        "--validation-physics-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs_burgers_dataset")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--loss-balance",
        dest="loss_balance",
        action="store_true",
    )
    parser.add_argument(
        "--no-loss-balance",
        dest="loss_balance",
        action="store_false",
    )
    parser.add_argument(
        "--manifold-cnn",
        dest="manifold_cnn",
        action="store_true",
    )
    parser.add_argument(
        "--no-manifold-cnn",
        dest="manifold_cnn",
        action="store_false",
    )
    parser.add_argument(
        "--manifold-cnn-res",
        type=int,
        default=None,
        help="replace the latent->CNN branch with N residual blocks",
    )
    parser.add_argument("--manifold-cnn-width", type=int, default=None)
    parser.add_argument(
        "--cnn-res-blocks",
        type=int,
        default=None,
        help="replace the input tokenizer CNN with N residual blocks",
    )
    parser.add_argument("--cnn-width", type=int, default=None)
    parser.add_argument(
        "--distill-fno",
        type=str,
        default=None,
        help="npz with FNO teacher fields [n, nx, nt] for the training split",
    )
    parser.add_argument("--distill-weight", type=float, default=None)
    parser.add_argument("--distill-points", type=int, default=None)
    parser.add_argument(
        "--query-local-features",
        dest="query_local_features",
        action="store_true",
    )
    parser.add_argument(
        "--no-query-local-features",
        dest="query_local_features",
        action="store_false",
    )
    parser.add_argument(
        "--time-conditioned-context",
        dest="time_conditioned_context",
        action="store_true",
    )
    parser.add_argument(
        "--no-time-conditioned-context",
        dest="time_conditioned_context",
        action="store_false",
    )
    parser.set_defaults(
        loss_balance=None,
        manifold_cnn=None,
        query_local_features=None,
        time_conditioned_context=None,
        use_cross_attention=None,
    )
    args = parser.parse_args()

    cfg = DatasetConfig(seed=args.seed)
    if args.preset == "stable":
        cfg.manifold_hidden = 64
        cfg.latent_dim = 16
        cfg.cnn_channels = 16
        cfg.token_count = 12
        cfg.d_model = 32
        cfg.n_layers = 2
        cfg.data_points_per_sample = 8
        cfg.physics_points_per_sample = 2
        cfg.initial_points_per_sample = 8
        cfg.boundary_points_per_sample = 1
        cfg.manifold_steps = 300
        cfg.steps = 1000
        cfg.lr = 5.0e-4
        cfg.grad_clip_norm = 1.0
        cfg.min_lr_ratio = 0.02
        cfg.warmup_steps = 150
        cfg.physics_weight = 0.05
    elif args.preset == "balanced":
        cfg.train_samples = 512
        cfg.data_points_per_sample = 32
        cfg.initial_points_per_sample = 16
        cfg.manifold_steps = 500
        cfg.steps = 6000
        cfg.lr_decay_steps = 4000
        cfg.lr = 5.0e-4
        cfg.min_lr_ratio = 0.02
        cfg.warmup_steps = 200
        cfg.eval_every = 200
        cfg.early_stop_patience = 8
    elif args.preset == "capacity":
        cfg.manifold_hidden = 128
        cfg.latent_dim = 32
        cfg.cnn_channels = 32
        cfg.token_count = 16
        cfg.d_model = 64
        cfg.n_layers = 3
        cfg.data_points_per_sample = 16
        cfg.physics_points_per_sample = 6
        cfg.initial_points_per_sample = 8
        cfg.boundary_points_per_sample = 2
        cfg.manifold_steps = 500
        cfg.steps = 1500
        cfg.lr = 5.0e-4
        cfg.grad_clip_norm = 1.0
        cfg.min_lr_ratio = 0.05
        cfg.warmup_steps = 200
        cfg.physics_weight = 0.10
    elif args.preset == "wide":
        cfg.train_samples = 512
        cfg.latent_dim = 32
        cfg.manifold_hidden = 256
        cfg.cnn_channels = 32
        cfg.token_count = 32
        cfg.d_model = 64
        cfg.n_layers = 3
        cfg.data_points_per_sample = 32
        cfg.physics_points_per_sample = 4
        cfg.initial_points_per_sample = 16
        cfg.manifold_steps = 500
        cfg.steps = 12000
        cfg.lr_decay_steps = 12000
        cfg.lr = 1.0e-3
        cfg.grad_clip_norm = 1.0
        cfg.min_lr_ratio = 0.02
        cfg.warmup_steps = 200
        cfg.physics_weight = 0.05
        cfg.eval_every = 500
        cfg.early_stop_patience = 16
        cfg.query_local_features = True
    if args.steps is not None:
        cfg.steps = args.steps
    if args.lr_decay_steps is not None:
        cfg.lr_decay_steps = args.lr_decay_steps
    if args.manifold_steps is not None:
        cfg.manifold_steps = args.manifold_steps
    if args.train_samples is not None:
        cfg.train_samples = args.train_samples
    if args.dev_samples is not None:
        cfg.dev_samples = args.dev_samples
    if args.test_samples is not None:
        cfg.test_samples = args.test_samples
    if args.eval_samples is not None:
        cfg.eval_samples = args.eval_samples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.data_points is not None:
        cfg.data_points_per_sample = args.data_points
    if args.physics_points is not None:
        cfg.physics_points_per_sample = args.physics_points
    if args.initial_points is not None:
        cfg.initial_points_per_sample = args.initial_points
    if args.boundary_points is not None:
        cfg.boundary_points_per_sample = args.boundary_points
    if args.initial_patch_radius is not None:
        cfg.initial_patch_radius = args.initial_patch_radius
    if args.physics_slope_bias is not None:
        cfg.physics_slope_bias = args.physics_slope_bias
    if args.data_gradient_weight is not None:
        cfg.data_gradient_weight = args.data_gradient_weight
    if args.periodic_derivative_weight is not None:
        cfg.periodic_derivative_weight = args.periodic_derivative_weight
    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim
    if args.manifold_hidden is not None:
        cfg.manifold_hidden = args.manifold_hidden
    if args.manifold_arch is not None:
        cfg.manifold_arch = args.manifold_arch
    if args.conv_ae_channels is not None:
        cfg.conv_ae_channels = args.conv_ae_channels
    if args.denoise_sigma is not None:
        cfg.denoise_sigma = args.denoise_sigma
    if args.denoise_weight is not None:
        cfg.denoise_weight = args.denoise_weight
    if args.denoise_steps is not None:
        cfg.denoise_steps = args.denoise_steps
    if args.denoise_rate is not None:
        cfg.denoise_rate = args.denoise_rate
    if args.denoise_eval_sigma is not None:
        cfg.denoise_eval_sigma = args.denoise_eval_sigma
    if args.denoise_until is not None:
        cfg.denoise_until = args.denoise_until
    if args.embed_flow:
        cfg.embed_flow = True
    if args.embed_flow_max_sigma is not None:
        cfg.embed_flow_max_sigma = args.embed_flow_max_sigma
    if args.embed_flow_weight is not None:
        cfg.embed_flow_weight = args.embed_flow_weight
    if cfg.embed_flow and cfg.denoise_weight == 0.0:
        cfg.denoise_weight = cfg.embed_flow_weight
    if args.use_cross_attention is not None:
        cfg.use_cross_attention = args.use_cross_attention
    if args.tokenizer is not None:
        cfg.tokenizer = args.tokenizer
    if args.arch is not None:
        cfg.arch = args.arch
    if args.latent_channels is not None:
        cfg.latent_channels = args.latent_channels
    if args.latent_levels is not None:
        cfg.latent_levels = args.latent_levels
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if cfg.arch == "cnn_only":
        cfg.query_local_features = True
        cfg.latent_ode = False
    if cfg.arch == "rollout":
        cfg.query_local_features = True
        cfg.latent_ode = False
    if cfg.arch == "operator":
        cfg.query_local_features = False
    if cfg.arch == "latent_grid":
        cfg.query_local_features = False
    if args.fourier_bands is not None:
        cfg.fourier_bands = args.fourier_bands
    if args.time_fourier_bands is not None:
        cfg.time_fourier_bands = args.time_fourier_bands
    if args.head_width is not None:
        cfg.head_width = args.head_width
    if args.query_local_window is not None:
        cfg.query_local_window = args.query_local_window
    if args.decoder is not None:
        cfg.decoder = args.decoder
    if args.grid_channels is not None:
        cfg.grid_channels = args.grid_channels
    if args.grid_layers is not None:
        cfg.grid_layers = args.grid_layers
    if args.grid_seed is not None:
        cfg.grid_seed = args.grid_seed
    if args.spectral_modes_x is not None:
        cfg.spectral_modes_x = args.spectral_modes_x
    if args.spectral_modes_t is not None:
        cfg.spectral_modes_t = args.spectral_modes_t
    if args.latent_ode:
        cfg.latent_ode = True
    if args.ode_steps is not None:
        cfg.ode_steps = args.ode_steps
    if args.ode_hidden is not None:
        cfg.ode_hidden = args.ode_hidden
    if args.temporal is not None:
        cfg.temporal = args.temporal
    if args.rollout_steps is not None:
        cfg.rollout_steps = args.rollout_steps
    if args.rollout_teacher_forcing is not None:
        cfg.rollout_teacher_forcing = args.rollout_teacher_forcing
    if args.rollout_delta:
        cfg.rollout_delta = True
    if args.operator_attention:
        cfg.operator_attention = True
    if args.cnn_channels is not None:
        cfg.cnn_channels = args.cnn_channels
    if args.token_count is not None:
        cfg.token_count = args.token_count
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.n_heads is not None:
        cfg.n_heads = args.n_heads
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.grad_clip_norm is not None:
        cfg.grad_clip_norm = args.grad_clip_norm
    if args.min_lr_ratio is not None:
        cfg.min_lr_ratio = args.min_lr_ratio
    if args.loss_floor is not None:
        cfg.loss_floor = args.loss_floor
    if args.loss_balance_warmup is not None:
        cfg.loss_balance_warmup = args.loss_balance_warmup
    if args.physics_weight is not None:
        cfg.physics_weight = args.physics_weight
    if args.nu is not None:
        cfg.nu = args.nu
    if args.physics_nonlinear_scale is not None:
        cfg.physics_nonlinear_scale = args.physics_nonlinear_scale
    if args.validation_physics_weight is not None:
        cfg.validation_physics_weight = args.validation_physics_weight
    if args.loss_balance is not None:
        cfg.loss_balance = args.loss_balance
    if args.manifold_cnn is not None:
        cfg.use_manifold_cnn = args.manifold_cnn
    if args.manifold_cnn_res is not None:
        cfg.manifold_cnn_res_blocks = args.manifold_cnn_res
        if args.manifold_cnn_res > 0:
            cfg.use_manifold_cnn = True
    if args.manifold_cnn_width is not None:
        cfg.manifold_cnn_width = args.manifold_cnn_width
    if args.cnn_res_blocks is not None:
        cfg.cnn_res_blocks = args.cnn_res_blocks
    if args.cnn_width is not None:
        cfg.cnn_width = args.cnn_width
    if args.distill_fno is not None:
        cfg.distill_fno_path = args.distill_fno
    if args.distill_weight is not None:
        cfg.distill_weight = args.distill_weight
    if args.distill_points is not None:
        cfg.distill_points_per_sample = args.distill_points
    if args.query_local_features is not None:
        cfg.query_local_features = args.query_local_features
    if args.time_conditioned_context is not None:
        cfg.time_conditioned_context = args.time_conditioned_context
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg.early_stop_min_delta = args.early_stop_min_delta
    if cfg.grad_clip_norm <= 0.0:
        parser.error("--grad-clip-norm must be positive")
    if cfg.manifold_hidden <= 0:
        parser.error("--manifold-hidden must be positive")
    if cfg.conv_ae_channels <= 0:
        parser.error("--conv-ae-channels must be positive")
    if cfg.denoise_sigma < 0.0:
        parser.error("--denoise-sigma must be non-negative")
    if cfg.denoise_weight < 0.0:
        parser.error("--denoise-weight must be non-negative")
    if cfg.denoise_steps < 0:
        parser.error("--denoise-steps must be non-negative")
    if cfg.denoise_rate < 0.0:
        parser.error("--denoise-rate must be non-negative")
    if cfg.denoise_eval_sigma < 0.0:
        parser.error("--denoise-eval-sigma must be non-negative")
    if cfg.embed_flow_max_sigma <= 0.0:
        parser.error("--embed-flow-max-sigma must be positive")
    if cfg.embed_flow_weight < 0.0:
        parser.error("--embed-flow-weight must be non-negative")
    if cfg.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if not 0.0 <= cfg.rollout_teacher_forcing <= 1.0:
        parser.error("--rollout-teacher-forcing must be in [0, 1]")
    if cfg.eval_samples <= 0:
        parser.error("--eval-samples must be positive")
    if not 0.0 <= cfg.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if cfg.fourier_bands <= 0:
        parser.error("--fourier-bands must be positive")
    if cfg.time_fourier_bands <= 0:
        parser.error("--time-fourier-bands must be positive")
    if cfg.head_width <= 0:
        parser.error("--head-width must be positive")
    if cfg.query_local_window < 0:
        parser.error("--query-local-window must be non-negative")
    if cfg.grid_channels <= 0:
        parser.error("--grid-channels must be positive")
    if cfg.grid_layers <= 0:
        parser.error("--grid-layers must be positive")
    if cfg.grid_seed <= 0:
        parser.error("--grid-seed must be positive")
    if cfg.spectral_modes_x <= 0 or cfg.spectral_modes_t <= 0:
        parser.error("--spectral-modes-x / --spectral-modes-t must be positive")
    if cfg.ode_steps <= 0:
        parser.error("--ode-steps must be positive")
    if cfg.ode_hidden <= 0:
        parser.error("--ode-hidden must be positive")
    if cfg.data_points_per_sample <= 0:
        parser.error("--data-points must be positive")
    if cfg.physics_points_per_sample <= 0:
        parser.error("--physics-points must be positive")
    if cfg.initial_points_per_sample <= 0:
        parser.error("--initial-points must be positive")
    if cfg.boundary_points_per_sample <= 0:
        parser.error("--boundary-points must be positive")
    if cfg.initial_patch_radius < 0:
        parser.error("--initial-patch-radius must be non-negative")
    if cfg.physics_slope_bias < 0.0:
        parser.error("--physics-slope-bias must be non-negative")
    if cfg.data_gradient_weight < 0.0:
        parser.error("--data-gradient-weight must be non-negative")
    if cfg.periodic_derivative_weight < 0.0:
        parser.error("--periodic-derivative-weight must be non-negative")
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be in [0, 1]")
    if cfg.loss_floor <= 0.0:
        parser.error("--loss-floor must be positive")
    if cfg.loss_balance_warmup < 0:
        parser.error("--loss-balance-warmup must be non-negative")
    if cfg.validation_physics_weight < 0.0:
        parser.error("--validation-physics-weight must be non-negative")
    if cfg.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if cfg.early_stop_min_delta < 0.0:
        parser.error("--early-stop-min-delta must be non-negative")
    if cfg.lr_decay_steps < 0:
        parser.error("--lr-decay-steps must be non-negative")
    if cfg.d_model % cfg.n_heads != 0:
        parser.error("--d-model must be divisible by --n-heads")
    if args.quick:
        cfg.steps = min(cfg.steps, 60)
        cfg.manifold_steps = min(cfg.manifold_steps, 40)
        cfg.eval_every = 20
        cfg.log_every = 10
        cfg.eval_samples = min(cfg.eval_samples, 16)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(device)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)

    print(f"device: {device}")
    print(f"preset: {args.preset}")
    print("loading PDEBench 1D Burgers Nu=0.01")
    started = time.perf_counter()
    train_fields = load_split(
        data_dir / cfg.train_file,
        cfg.train_samples,
    ).to(device)
    dev_fields = load_split(
        data_dir / cfg.dev_file,
        cfg.dev_samples,
    ).to(device)
    test_fields = load_split(
        data_dir / cfg.test_file,
        cfg.test_samples,
    ).to(device)
    if cfg.distill_fno_path:
        teacher_payload = np.load(cfg.distill_fno_path)
        teacher_all = torch.from_numpy(
            teacher_payload["fields"]
        ).float()
        if teacher_all.size(0) < train_fields.size(0):
            raise SystemExit(
                f"teacher fields {tuple(teacher_all.shape)} shorter than "
                f"train split {train_fields.size(0)}"
            )
        cfg._distill_fields = teacher_all[: train_fields.size(0)].to(device)
        print(
            "distillation teacher:",
            cfg.distill_fno_path,
            "fields", tuple(teacher_all.shape),
            "weight", cfg.distill_weight,
            "points", cfg.distill_points_per_sample,
        )
    print(
        "shapes:",
        tuple(train_fields.shape),
        tuple(dev_fields.shape),
        tuple(test_fields.shape),
    )
    print(
        "u range:",
        f"train [{float(train_fields.min()):.4f}, {float(train_fields.max()):.4f}]",
        f"test [{float(test_fields.min()):.4f}, {float(test_fields.max()):.4f}]",
    )

    if cfg.arch == "latent_grid":
        model = BurgersStructuredLatentPINN(cfg).to(device)
    elif cfg.arch == "operator":
        model = BurgersOperatorPINN(cfg).to(device)
    elif cfg.arch == "rollout":
        model = BurgersRolloutPINN(cfg).to(device)
    else:
        model = BurgersHybridPINN(cfg).to(device)
    parameter_counts = parameter_breakdown(model)
    print(f"model parameters: {count_parameters(model)}")
    print("parameter breakdown:", parameter_counts)
    LossScaleBalancer.default_warmup = cfg.loss_balance_warmup
    if cfg.loss_balance and cfg.loss_balance_warmup > 0:
        print(
            f"loss balancing warmup: {cfg.loss_balance_warmup} steps "
            "(references are frozen only after the model leaves its "
            "degenerate initial state)"
        )

    print("\n[1/2] pretrain manifold encoder")
    if model.manifold is None:
        manifold_history = []
        print(
            f"skipped: arch '{cfg.arch}' has no manifold autoencoder "
            "(the latent is produced otherwise)"
        )
    else:
        manifold_history = pretrain_manifold(
            model,
            cfg,
            train_fields,
            device,
        )
    if manifold_history:
        print(
            "manifold reconstruction MSE:",
            f"{manifold_history[-1]['manifold_loss']:.6e}",
        )
        print(
            "manifold max grad norm:",
            f"{manifold_history[-1]['max_grad_norm']:.4e}",
            f"skipped steps: {int(manifold_history[-1]['skipped_steps'])}",
        )

    print("\n[2/2] train hybrid PINN with train/dev split")
    history, best_state, stability = train(
        model,
        cfg,
        train_fields,
        dev_fields,
        device,
    )

    print("\nevaluating train/dev/test trajectories")
    train_metrics = evaluate_split(
        model,
        train_fields,
        min(cfg.eval_samples, train_fields.size(0)),
        cfg,
        device,
    )
    dev_metrics = evaluate_split(
        model,
        dev_fields,
        min(cfg.eval_samples, dev_fields.size(0)),
        cfg,
        device,
    )
    test_metrics = evaluate_split(
        model,
        test_fields,
        min(cfg.eval_samples, test_fields.size(0)),
        cfg,
        device,
    )
    gap = test_metrics["relative_l2"] - train_metrics["relative_l2"]
    ratio = (
        test_metrics["relative_l2"]
        / max(train_metrics["relative_l2"], 1.0e-8)
    )
    diagnosis = diagnose(
        train_metrics["relative_l2"],
        test_metrics["relative_l2"],
    )
    elapsed = time.perf_counter() - started

    payload = {
        "preset": args.preset,
        "config": asdict(cfg),
        "parameter_count": count_parameters(model),
        "parameter_breakdown": parameter_counts,
        "training_seconds": elapsed,
        "train": train_metrics,
        "dev": dev_metrics,
        "test": test_metrics,
        "stability": stability,
        "gap": {
            "test_minus_train_relative_l2": gap,
            "test_over_train_ratio": ratio,
            "diagnosis": diagnosis,
        },
        "manifold_history": manifold_history,
        "training_history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if args.save_model:
        torch.save(
            {
                "state_dict": best_state,
                "config": asdict(cfg),
                "parameter_count": count_parameters(model),
            },
            output_dir / "model.pt",
        )

    plot_training(history, output_dir / "training_curve.png")
    plot_prediction(
        model,
        test_fields[0],
        cfg,
        device,
        output_dir / "test_prediction.png",
    )

    print("\nresults")
    print(
        f"train relative L2: {train_metrics['relative_l2']:.4e} "
        f"PDE RMS: {train_metrics['pde_rms']:.4e}"
    )
    print(
        f"dev   relative L2: {dev_metrics['relative_l2']:.4e} "
        f"PDE RMS: {dev_metrics['pde_rms']:.4e}"
    )
    print(
        f"test  relative L2: {test_metrics['relative_l2']:.4e} "
        f"PDE RMS: {test_metrics['pde_rms']:.4e}"
    )
    print(f"test - train gap: {gap:.4e}")
    print(f"test / train ratio: {ratio:.3f}")
    print(f"diagnosis: {diagnosis}")
    print(
        "loss balancing:",
        f"enabled={cfg.loss_balance}",
        f"floor={cfg.loss_floor}",
    )
    print(
        "gradient stability:",
        f"max={stability['max_grad_norm']:.3e}",
        f"mean={stability['mean_grad_norm']:.3e}",
        f"clipped_steps={int(stability['clipped_steps'])}",
        f"skipped_steps={int(stability['skipped_steps'])}",
    )
    print(
        "best validation:",
        f"step={int(stability['best_dev_step'])}",
        f"score={float(stability['best_dev_score']):.4e}",
    )
    print(
        "early stopping:",
        f"stopped={stability['stopped_early']}",
        f"completed_steps={int(stability['completed_steps'])}",
        f"stale_evaluations={int(stability['stale_evaluations'])}",
    )
    print(f"training seconds: {elapsed:.2f}")
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(
            f"peak GPU memory: allocated {allocated:.2f} GiB, "
            f"reserved {reserved:.2f} GiB"
        )
    print(f"artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
