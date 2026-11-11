"""PB_OOD 在独立 dev 集上的干净对照：只改变 d_model / n_layers。"""
import subprocess
import sys

COMMON = [
    "--preset", "wide",
    "--data-dir", "datasets/burgers_custom_pb_ood",
    "--train-file", "burgers_nu0.01_train_split.parquet",
    "--dev-file", "burgers_nu0.01_dev_split.parquet",
    "--test-file", "burgers_nu0.01_test.parquet",
    "--seed", "123",
    "--dropout", "0.05",
    "--physics-weight", "0.05",
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--lr", "0.001", "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500",
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

RUNS = [
    ("pb_ood_clean_d64_l3", ["--d-model", "64", "--n-layers", "3"]),
    ("pb_ood_clean_d96_l4", ["--d-model", "96", "--n-layers", "4"]),
]

if __name__ == "__main__":
    failed = []
    for name, extra in RUNS:
        print(f"\n===== {name} =====", flush=True)
        cmd = [sys.executable, "code/demo_burgers_dataset.py", *COMMON, *extra,
               "--output-dir", f"experiments/outputs_burgers_dataset/{name}"]
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print("\nfailed:", failed or "none")
    sys.exit(1 if failed else 0)
