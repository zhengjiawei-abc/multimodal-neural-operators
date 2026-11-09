from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_burgers_dataset import (  # noqa: E402
    BurgersHybridPINN,
    BurgersOperatorPINN,
    BurgersRolloutPINN,
    DatasetConfig,
    grid_pde_residual,
    load_split,
    query_local_features,
    sample_data_batch,
)


def build_config(raw: Dict[str, object]) -> DatasetConfig:
    known = {field.name for field in fields(DatasetConfig)}
    return DatasetConfig(**{k: v for k, v in raw.items() if k in known})


def load_run(run_dir: Path, data_dir: Path) -> Tuple[BurgersHybridPINN, DatasetConfig, Dict]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = build_config(payload["config"])
    cfg.data_dir = str(data_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    if cfg.arch == "rollout":
        model = BurgersRolloutPINN(cfg)
    elif cfg.arch == "operator":
        model = BurgersOperatorPINN(cfg)
    else:
        model = BurgersHybridPINN(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model, cfg, payload


def predict_field(
    model: BurgersHybridPINN,
    field: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    u0 = field[:, 0, 0].unsqueeze(0)
    if cfg.decoder == "grid":
        with torch.no_grad():
            _, _, prediction = model.decode_grid(u0)
        return prediction[0]
    x_axis = torch.linspace(0.0, 1.0, cfg.nx, device=device)
    t_axis = torch.linspace(0.0, 1.0, cfg.nt, device=device)
    x_flat = x_axis.repeat_interleave(cfg.nt).unsqueeze(1)
    t_flat = t_axis.repeat(cfg.nx).unsqueeze(1)
    latent, context = model.encode(u0)
    point_count = x_flat.size(0)
    chunks: List[torch.Tensor] = []
    for start in range(0, point_count, 4096):
        stop = min(start + 4096, point_count)
        with torch.no_grad():
            chunks.append(
                model.decode_queries(
                    context,
                    latent.expand(stop - start, -1),
                    x_flat[start:stop],
                    t_flat[start:stop],
                    query_local_features(
                        model,
                        u0.expand(stop - start, -1),
                        x_flat[start:stop],
                        cfg.initial_patch_radius,
                    ),
                )
            )
    return torch.cat(chunks, dim=0).reshape(cfg.nx, cfg.nt)


def encode_latents(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    model.eval()
    codes: List[torch.Tensor] = []
    for start in range(0, fields.size(0), batch_size):
        stop = min(start + batch_size, fields.size(0))
        with torch.no_grad():
            latent = model.manifold.encode(fields[start:stop, :, 0, 0])
        codes.append(latent.cpu())
    return torch.cat(codes, dim=0)


def plot_high_dimension(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    sample = fields[0]
    prediction = predict_field(model, sample, cfg, device).cpu().numpy()
    truth = sample[:, :, 0].cpu().numpy()
    if model.manifold is None:
        u0 = sample[:, 0, 0].cpu().numpy()
        reconstruction = np.full_like(u0, np.nan)
    else:
        u0_tensor = sample[:, 0, 0].unsqueeze(0)
        with torch.no_grad():
            latent = model.manifold.encode(u0_tensor)
            reconstruction = model.manifold.decode(latent)[0].cpu().numpy()
        u0 = u0_tensor[0].cpu().numpy()

    figure, axes = plt.subplots(1, 4, figsize=(19, 3.8))
    for axis, data, title in (
        (axes[0], truth, "dataset u(x, t)"),
        (axes[1], prediction, "model u(x, t)"),
        (axes[2], prediction - truth, "model - dataset"),
    ):
        image = axis.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis" if title != "model - dataset" else "coolwarm",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        figure.colorbar(image, ax=axis)
    axes[0].set_ylabel("t")
    axes[3].plot(np.linspace(0.0, 1.0, cfg.nx), u0, label="u0 (data)")
    axes[3].plot(
        np.linspace(0.0, 1.0, cfg.nx),
        reconstruction,
        linestyle="--",
        label=f"manifold decode ({cfg.latent_dim}-d)",
    )
    axes[3].set_title("high-dim u0 vs low-dim decode")
    axes[3].set_xlabel("x")
    axes[3].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_low_dimension(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    if model.manifold is None:
        print("skip low-dim figure: cnn_only run has no manifold encoder")
        return
    latents = encode_latents(model, fields, device).numpy()
    centered = latents - latents.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / max(1, latents.shape[0] - 1)
    explained = variance / variance.sum()
    projection = centered @ vt[:2].T
    slope = np.abs(np.diff(fields[:, :, 0, 0].cpu().numpy(), axis=1)).max(axis=1)
    with torch.no_grad():
        reconstruction = model.manifold.decode(
            torch.from_numpy(latents).to(device)
        ).cpu()
    target = fields[:, :, 0, 0].cpu()
    relative_error = (
        torch.linalg.vector_norm(reconstruction - target, dim=1)
        / torch.linalg.vector_norm(target, dim=1).clamp_min(1.0e-8)
    ).numpy()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(
        np.arange(1, len(explained) + 1),
        explained * 100.0,
    )
    axes[0].plot(
        np.arange(1, len(explained) + 1),
        np.cumsum(explained) * 100.0,
        color="black",
        marker="o",
        markersize=3,
        label="cumulative",
    )
    axes[0].set_title(f"latent PCA spectrum ({cfg.latent_dim}-d codes)")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("explained variance [%]")
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(
        projection[:, 0],
        projection[:, 1],
        c=slope,
        s=8,
        cmap="magma",
    )
    axes[1].set_title("latent space (PCA of codes)")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    figure.colorbar(scatter, ax=axes[1], label="max |du0/dx|")

    axes[2].hist(relative_error, bins=30, color="steelblue")
    axes[2].set_title("manifold reconstruction error")
    axes[2].set_xlabel("relative L2 of decoded u0")
    axes[2].set_ylabel("trajectories")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_gradients(
    model: BurgersHybridPINN,
    cfg: DatasetConfig,
    payload: Dict,
    train_fields: torch.Tensor,
    device: torch.device,
    path: Path,
) -> None:
    history = payload.get("training_history", [])
    steps = [row["step"] for row in history]

    model.train()
    if cfg.decoder == "grid":
        indices = torch.randint(
            0, train_fields.size(0), (cfg.batch_size,), device=device
        )
        u0 = train_fields[indices, :, 0, 0]
        target = train_fields[indices, :, :, 0]
        _, _, prediction = model.decode_grid(u0)
        loss = F.mse_loss(prediction, target)
    else:
        u0, x, t, target, _ = sample_data_batch(
            train_fields,
            cfg,
            device,
            points_per_sample=cfg.data_points_per_sample,
        )
        points = cfg.data_points_per_sample
        latent, context = model.encode(u0)
        prediction = model.decode_queries(
            context.repeat_interleave(points, dim=0),
            latent.repeat_interleave(points, dim=0),
            x,
            t,
            query_local_features(
                model,
                u0.repeat_interleave(points, dim=0),
                x,
                cfg.initial_patch_radius,
            ),
        )
        loss = F.mse_loss(prediction, target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    module_names: List[str] = []
    module_norms: List[float] = []
    for name, module in model.named_children():
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        module_names.append(name)
        module_norms.append(np.sqrt(total))
    model.zero_grad(set_to_none=True)

    figure, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].plot(steps, [row["grad_norm"] for row in history], color="crimson")
    axes[0].set_yscale("log")
    axes[0].set_title("gradient norm during training")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("grad norm (pre-clip)")
    twin = axes[0].twinx()
    twin.plot(steps, [row["learning_rate"] for row in history], color="gray", alpha=0.6)
    twin.set_ylabel("learning rate", color="gray")

    for key in ("data", "physics", "initial", "boundary", "reconstruction"):
        if key in history[0]:
            axes[1].plot(steps, [row[key] for row in history], label=key)
    axes[1].set_yscale("log")
    axes[1].set_title("loss terms")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=8)

    axes[2].barh(module_names, module_norms, color="slateblue")
    axes[2].set_xscale("log")
    axes[2].set_title("gradient norm per module (data loss)")
    axes[2].set_xlabel("grad norm")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_rollout_error(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    """Error growth along the autoregressive rollout (rollout arch only)."""
    count = min(8, fields.size(0))
    sample = fields[:count]
    u0 = sample[:, :, 0, 0]
    with torch.no_grad():
        rolled = model.march(u0, cfg.nt - 1, 0)
    prediction = rolled.permute(1, 2, 0)
    target = sample[:, :, :, 0]
    steps = np.arange(1, cfg.nt)
    per_step = []
    first_step = []
    for index in range(1, cfg.nt):
        difference = prediction[:, :, index] - target[:, :, index]
        reference = torch.linalg.vector_norm(
            target[:, :, index], dim=1
        ).clamp_min(1.0e-8)
        errors = (
            torch.linalg.vector_norm(difference, dim=1) / reference
        ).cpu().numpy()
        per_step.append(float(np.mean(errors)))
        first_step.append(float(errors[0]))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(steps, per_step, label="mean over samples")
    axes[0].plot(steps, first_step, alpha=0.5, label="sample 0")
    axes[0].set_yscale("log")
    axes[0].set_title("rollout error vs step")
    axes[0].set_xlabel("rollout step")
    axes[0].set_ylabel("relative L2 at that time slice")
    axes[0].legend(fontsize=8)

    axes[1].imshow(
        (prediction[0] - target[0]).cpu().numpy().T,
        origin="lower",
        aspect="auto",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="coolwarm",
    )
    axes[1].set_title("rollout - dataset (sample 0)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    axes[2].plot(
        np.linspace(0.0, 1.0, cfg.nx),
        target[0, :, -1].cpu().numpy(),
        color="black",
        label="dataset t=1",
    )
    axes[2].plot(
        np.linspace(0.0, 1.0, cfg.nx),
        prediction[0, :, -1].cpu().numpy(),
        linestyle="--",
        label="rollout t=1",
    )
    axes[2].set_title("final time slice")
    axes[2].set_xlabel("x")
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print("rollout error at step 1:", f"{per_step[0]:.4f}")
    print("rollout error at step 10:", f"{per_step[9]:.4f}")
    print("rollout error at step 100:", f"{per_step[-1]:.4f}")


def plot_residual(
    model: BurgersHybridPINN,
    fields: torch.Tensor,
    cfg: DatasetConfig,
    device: torch.device,
    path: Path,
) -> None:
    sample = fields[0]
    prediction = predict_field(model, sample, cfg, device).unsqueeze(0)
    residual = grid_pde_residual(prediction, cfg, cfg.nu)[0].cpu().numpy()
    truth = sample[:, :, 0].cpu().numpy()
    slope = np.gradient(truth, axis=0)
    residual_interior = residual
    slope_interior = np.abs(slope[1:-1, 1:-1])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    image = axes[0].imshow(
        residual_interior.T,
        origin="lower",
        aspect="auto",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="coolwarm",
    )
    axes[0].set_title("PDE residual r(x, t)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    figure.colorbar(image, ax=axes[0])

    axes[1].scatter(
        slope_interior.ravel(),
        np.abs(residual_interior).ravel(),
        s=2,
        alpha=0.25,
        color="darkorange",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("residual vs |du/dx|")
    axes[1].set_xlabel("|du/dx| of dataset")
    axes[1].set_ylabel("|r|")

    axes[2].plot(
        np.linspace(0.0, 1.0, residual_interior.shape[1]),
        np.sqrt((residual_interior**2).mean(axis=0)),
    )
    axes[2].set_title("residual RMS over time")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("RMS(r)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="datasets/pdebench_burgers")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, payload = load_run(run_dir, Path(args.data_dir))
    model.to(device)
    print(device, "| decoder", cfg.decoder, "| latent", cfg.latent_dim)
    fields = load_split(Path(cfg.data_dir) / cfg.test_file, 64)
    train_fields = load_split(Path(cfg.data_dir) / cfg.train_file, 128)

    plot_high_dimension(
        model,
        fields.to(device),
        cfg,
        device,
        output_dir / "analysis_high_dim.png",
    )
    plot_low_dimension(
        model,
        fields.to(device),
        cfg,
        device,
        output_dir / "analysis_low_dim.png",
    )
    plot_gradients(
        model,
        cfg,
        payload,
        train_fields.to(device),
        device,
        output_dir / "analysis_gradients.png",
    )
    plot_residual(
        model,
        fields.to(device),
        cfg,
        device,
        output_dir / "analysis_residual.png",
    )
    if cfg.arch == "rollout":
        plot_rollout_error(
            model,
            fields.to(device),
            cfg,
            device,
            output_dir / "analysis_rollout.png",
        )
    print("figures written to", output_dir.resolve())


if __name__ == "__main__":
    main()
