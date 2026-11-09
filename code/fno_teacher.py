"""Train an FNO teacher and export its full fields for PINN distillation.

FNO is itself a PDE solution operator ``u0 -> u(x,t)`` trained with full-field
supervision (see ``baselines_1d_burgers.py``).  This script trains it on the same
training split the hybrid PINN is about to see (``--train-samples``), then writes

  * ``fno_fields.npz`` -- key ``fields``: float16 array [n, nx, nt] in the same
    order as ``load_split(train_file, n)``, i.e. directly usable as dense
    pseudo-labels for ``--distill-fno``;
  * ``model.pt``       -- FNO weights;
  * ``metrics.json``   -- train/test relative L2, params, wall clock.

It is deliberately small: the recorded FNO baseline reaches test L2 0.1742
(512 samples) / 0.2304 (256 samples) in 4000 steps, i.e. under a minute on the
GPU, so a teacher can be produced on demand.

Usage:

    code\\.venv\\Scripts\\python.exe code\\fno_teacher.py --train-samples 512 \
        --output-dir experiments\\outputs_burgers_dataset\\lowdata\\fno_teacher_n512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from baselines_1d_burgers import (  # noqa: E402
    FNO1d,
    build_schedule,
    evaluate_fno,
    relative_l2,
)
from demo_burgers_dataset import (  # noqa: E402
    configure_backend,
    count_parameters,
    load_split,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "datasets" / "pdebench_burgers")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    configure_backend(device)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_fields = load_split(
        args.data_dir / "burgers_nu0.01_train_000.parquet",
        args.train_samples,
    ).to(device)
    test_fields = load_split(
        args.data_dir / "burgers_nu0.01_test.parquet",
        args.test_samples,
    ).to(device)
    nx, nt = train_fields.size(1), train_fields.size(2)
    print(f"device={device} train={tuple(train_fields.shape)}"
          f" test={tuple(test_fields.shape)}")

    model = FNO1d(args.modes, args.width, args.layers, nt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_schedule(optimizer, args.lr, args.steps, args.warmup)
    started = time.time()
    for step in range(args.steps):
        indices = torch.randint(0, train_fields.size(0),
                                (args.batch_size,), device=device)
        u0 = train_fields[indices, :, 0, 0]
        target = train_fields[indices, :, :, 0]
        prediction = model(u0)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:>6}  loss {float(loss.detach()):.6f}",
                  flush=True)
    seconds = time.time() - started

    train_l2 = evaluate_fno(model, train_fields, device, args.batch_size)
    test_l2 = evaluate_fno(model, test_fields, device, args.batch_size)
    parameters = count_parameters(model)
    print(f"train_L2={train_l2:.4f} test_L2={test_l2:.4f}"
          f" params={parameters} seconds={seconds:.1f}")

    # Export the full-field teacher for distillation, in training-split order.
    teacher = []
    model.eval()
    with torch.no_grad():
        for start in range(0, train_fields.size(0), 64):
            u0 = train_fields[start:start + 64, :, 0, 0]
            teacher.append(model(u0).cpu().numpy().astype(np.float16))
    teacher_fields = np.concatenate(teacher, axis=0)
    np.savez_compressed(args.output_dir / "fno_fields.npz",
                        fields=teacher_fields)
    torch.save({"state_dict": model.state_dict()},
               args.output_dir / "model.pt")
    metrics = {
        "model": "fno",
        "config": vars(args) | {"output_dir": str(args.output_dir),
                                "device": str(device)},
        "parameter_count": parameters,
        "train_relative_l2": train_l2,
        "test_relative_l2": test_l2,
        "training_seconds": seconds,
        "teacher_fields_shape": list(teacher_fields.shape),
        "teacher_fields_dtype": str(teacher_fields.dtype),
    }
    metrics["config"].pop("data_dir", None)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
