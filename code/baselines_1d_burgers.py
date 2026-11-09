"""External operator-learning baselines on the 1D Burgers task.

The point of this script is to answer a question the hybrid model cannot answer
from its own ablations: is it actually good, or merely better than its own
stripped-down variants? Two standard baselines are provided, both trained on the
same PDEBench splits and scored with the same relative-L2 metric:

* ``fno``      -- Fourier Neural Operator mapping u0 -> u(x, t) on the grid.
* ``deeponet`` -- branch/trunk network mapping (u0, x, t) -> u(x, t).

A time-mean predictor is also reported as a trivial floor.

Fairness note: FNO is trained with full-field supervision (its standard
training recipe), while the hybrid model only sees a handful of sampled points
per trajectory per step. If FNO wins, the honest reading is that the hybrid's
sparse-supervision setting - not its architecture - is what costs accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from demo_burgers_dataset import (
    configure_backend,
    count_parameters,
    load_split,
    set_seed,
)


class SpectralConv1d(nn.Module):
    """Spectral convolution over the spatial axis (the core FNO layer)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes,
                dtype=torch.cfloat,
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, length = features.shape
        spectrum = torch.fft.rfft(features)
        kept = min(self.modes, spectrum.size(-1))
        out_spectrum = torch.zeros(
            batch,
            self.out_channels,
            spectrum.size(-1),
            dtype=torch.cfloat,
            device=features.device,
        )
        out_spectrum[:, :, :kept] = torch.einsum(
            "bim,iom->bom",
            spectrum[:, :, :kept],
            self.weight[:, :, :kept],
        )
        return torch.fft.irfft(out_spectrum, n=length)


class FNO1d(nn.Module):
    """Maps an initial field u0(x) to the full space-time solution u(x, t)."""

    def __init__(
        self,
        modes: int,
        width: int,
        n_layers: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv1d(1, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.local = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, out_channels, kernel_size=1),
        )

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        features = self.lift(u0.unsqueeze(1))
        for spectral, local in zip(self.spectral, self.local):
            features = F.gelu(spectral(features) + local(features))
        field = self.project(features)
        return field.permute(0, 2, 1)


class DeepONet(nn.Module):
    """Branch (u0) x trunk (x, t) network with a shared coefficient basis."""

    def __init__(
        self,
        nx: int,
        basis: int,
        branch_hidden: int,
        trunk_hidden: int,
    ) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(nx, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, basis),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, basis),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        u0: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        branch = self.branch(u0)
        trunk = self.trunk(coords)
        return (branch.unsqueeze(1) * trunk).sum(dim=-1) + self.bias


def build_schedule(
    optimizer: torch.optim.Optimizer,
    lr: float,
    steps: int,
    warmup: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return 0.1 + 0.9 * (step + 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(steps - warmup, 1))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


@torch.no_grad()
def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    flat_prediction = prediction.reshape(prediction.size(0), -1)
    flat_target = target.reshape(target.size(0), -1)
    error = torch.linalg.vector_norm(
        flat_prediction - flat_target,
        dim=1,
    )
    scale = torch.linalg.vector_norm(flat_target, dim=1).clamp_min(1.0e-8)
    return float((error / scale).mean())


def evaluate_fno(
    model: FNO1d,
    fields: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    errors = []
    for start in range(0, fields.size(0), batch_size):
        u0 = fields[start : start + batch_size, :, 0, 0].to(device)
        target = fields[start : start + batch_size, :, :, 0].to(device)
        errors.append(relative_l2(model(u0), target))
    model.train()
    return float(np.mean(errors))


def grid_coordinates(
    nx: int,
    nt: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, device=device)
    t = torch.linspace(0.0, 1.0, nt, device=device)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=-1)


def evaluate_deeponet(
    model: DeepONet,
    fields: torch.Tensor,
    device: torch.device,
    chunk: int = 4096,
) -> float:
    model.eval()
    nx, nt = fields.size(1), fields.size(2)
    coords = grid_coordinates(nx, nt, device)
    errors = []
    with torch.no_grad():
        for index in range(fields.size(0)):
            u0 = fields[index : index + 1, :, 0, 0].to(device)
            target = fields[index, :, :, 0].to(device).reshape(1, -1)
            pieces = [
                model(u0, coords[start : start + chunk])
                for start in range(0, coords.size(0), chunk)
            ]
            prediction = torch.cat(pieces, dim=1)
            errors.append(relative_l2(prediction, target))
    model.train()
    return float(np.mean(errors))


def sample_queries(
    fields: torch.Tensor,
    indices: torch.Tensor,
    points: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nx, nt = fields.size(1), fields.size(2)
    x_index = torch.randint(0, nx, (indices.size(0), points), device=device)
    t_index = torch.randint(0, nt, (indices.size(0), points), device=device)
    u0 = fields[indices, :, 0, 0]
    target = fields[indices.view(-1, 1), x_index, t_index, 0]
    coords = torch.stack(
        [
            x_index.float() / (nx - 1),
            t_index.float() / (nt - 1),
        ],
        dim=-1,
    )
    return u0, coords, target


def train_fno(
    fields: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    nx, nt = fields.size(1), fields.size(2)
    model = FNO1d(args.modes, args.width, args.layers, nt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_schedule(optimizer, args.lr, args.steps, args.warmup)
    history = []
    started = time.time()
    for step in range(args.steps):
        indices = torch.randint(
            0, fields.size(0), (args.batch_size,), device=device
        )
        u0 = fields[indices, :, 0, 0]
        target = fields[indices, :, :, 0]
        prediction = model(u0)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"  step {step:>6}  loss {float(loss.detach()):.6f}")
    return {
        "model": model,
        "history": history,
        "seconds": time.time() - started,
    }


def train_deeponet(
    fields: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    nx = fields.size(1)
    model = DeepONet(
        nx,
        args.basis,
        args.branch_hidden,
        args.trunk_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_schedule(optimizer, args.lr, args.steps, args.warmup)
    history = []
    started = time.time()
    for step in range(args.steps):
        indices = torch.randint(
            0, fields.size(0), (args.batch_size,), device=device
        )
        u0, coords, target = sample_queries(
            fields, indices, args.query_points, device
        )
        prediction = model(u0, coords)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"  step {step:>6}  loss {float(loss.detach()):.6f}")
    return {
        "model": model,
        "history": history,
        "seconds": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FNO / DeepONet baselines on 1D Burgers.",
    )
    parser.add_argument("--model", choices=("fno", "deeponet"), default="fno")
    parser.add_argument("--data-dir", type=str, default="datasets/pdebench_burgers")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--basis", type=int, default=128)
    parser.add_argument("--branch-hidden", type=int, default=256)
    parser.add_argument("--trunk-hidden", type=int, default=128)
    parser.add_argument("--query-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_samples = 16
        args.test_samples = 8
        args.steps = 10
        args.batch_size = 4
        args.log_every = 5

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(device)
    print(f"device: {device}  model: {args.model}")

    data_dir = Path(args.data_dir)
    train_fields = load_split(
        data_dir / "burgers_nu0.01_train_000.parquet",
        args.train_samples,
    ).to(device)
    test_fields = load_split(
        data_dir / "burgers_nu0.01_test.parquet",
        args.test_samples,
    ).to(device)
    print(f"train {tuple(train_fields.shape)}  test {tuple(test_fields.shape)}")

    test_target = test_fields[..., 0]
    time_mean = test_target.mean(dim=2, keepdim=True).expand_as(test_target)
    floor = relative_l2(time_mean, test_target)
    print(f"trivial time-mean baseline: {floor:.4f}")

    if args.model == "fno":
        result = train_fno(train_fields, args, device)
        test_l2 = evaluate_fno(
            result["model"], test_fields, device, args.batch_size
        )
        train_l2 = evaluate_fno(
            result["model"], train_fields, device, args.batch_size
        )
    else:
        result = train_deeponet(train_fields, args, device)
        test_l2 = evaluate_deeponet(result["model"], test_fields, device)
        train_l2 = evaluate_deeponet(result["model"], train_fields, device)

    parameters = count_parameters(result["model"])
    print(
        f"model={args.model}  params={parameters}  "
        f"train_L2={train_l2:.4f}  test_L2={test_l2:.4f}  "
        f"seconds={result['seconds']:.1f}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": args.model,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "modes": args.modes,
            "width": args.width,
            "layers": args.layers,
            "basis": args.basis,
            "branch_hidden": args.branch_hidden,
            "trunk_hidden": args.trunk_hidden,
            "query_points": args.query_points,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "seed": args.seed,
        },
        "parameter_count": parameters,
        "train_relative_l2": train_l2,
        "test_relative_l2": test_l2,
        "trivial_time_mean_l2": floor,
        "training_seconds": result["seconds"],
        "history": result["history"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
