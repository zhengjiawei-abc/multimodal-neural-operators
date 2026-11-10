"""
Monitor v4 augmentation experiment status and generate summary report
"""
import json
import time
from pathlib import Path
from datetime import datetime

OUTPUT_BASE = Path("experiments/outputs_burgers_dataset")
V4_EXPERIMENTS = [
    "pb_ood_aug_v4_baseline",
    "pb_ood_aug_v4_physics",
    "pb_ood_aug_v4_seed42",
]

def check_experiment_status(exp_name):
    """Check if an experiment is complete and get its metrics"""
    exp_dir = OUTPUT_BASE / exp_name
    metrics_file = exp_dir / "metrics.json"

    if not exp_dir.exists():
        return None, "not_started"

    if not metrics_file.exists():
        checkpoint = exp_dir / "checkpoint.pt"
        if checkpoint.exists():
            return None, "running"
        return None, "incomplete"

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics, "complete"
    except:
        return None, "error"

def format_metrics(metrics):
    """Format metrics for display"""
    if not metrics:
        return "N/A"

    test_l2 = metrics.get('test', {}).get('relative_l2', metrics.get('test', {}).get('l2', 'N/A'))
    test_pde = metrics.get('test', {}).get('pde_rms', 'N/A')
    gap = metrics.get('gap', {}).get('test_over_train_ratio', 1.0)
    gap_pct = (gap - 1) * 100

    return {
        'test_l2': test_l2,
        'test_pde': test_pde,
        'gap_pct': gap_pct,
    }

def monitor_v4_experiments():
    """Monitor v4 experiment completion"""
    print("=" * 70)
    print("V4 Augmentation Experiment Monitor")
    print("=" * 70)

    statuses = {}
    check_count = 0
    max_checks = 360  # 30 minutes of checking (5 second intervals)

    while check_count < max_checks:
        check_count += 1
        all_complete = True

        print("[{}] Status Check #{}: {}".format(
            datetime.now().strftime("%H:%M:%S"),
            check_count,
            time.time() % 1 < 0.5 and "checking..." or "ready"
        ))

        for exp_name in V4_EXPERIMENTS:
            metrics, status = check_experiment_status(exp_name)
            statuses[exp_name] = (metrics, status)

            if status == "complete":
                fmt = format_metrics(metrics)
                print("  [DONE] {}: Test_L2={} Gap={:+.1f}%".format(
                    exp_name.replace("pb_ood_aug_v4_", ""),
                    fmt['test_l2'] if isinstance(fmt['test_l2'], str) else "{:.4f}".format(fmt['test_l2']),
                    fmt['gap_pct'] if isinstance(fmt, dict) else 0
                ))
            elif status == "running":
                print("  [...] {}: Training in progress".format(
                    exp_name.replace("pb_ood_aug_v4_", "")
                ))
                all_complete = False
            elif status == "not_started":
                print("  [ ] {}: Not started".format(
                    exp_name.replace("pb_ood_aug_v4_", "")
                ))
                all_complete = False
            else:
                print("  [?] {}: {}".format(
                    exp_name.replace("pb_ood_aug_v4_", ""),
                    status
                ))
                all_complete = False

        if all_complete and all(s[1] == "complete" for s in statuses.values()):
            print("\n" + "=" * 70)
            print("All V4 experiments completed!")
            print("=" * 70)
            generate_summary_report(statuses)
            break

        if check_count < max_checks:
            time.sleep(5)

    if check_count >= max_checks:
        print("\nMonitoring timeout (30 min). Some experiments may still be running.")
        print("You can check: experiments/outputs_burgers_dataset/pb_ood_aug_v4_*")

def generate_summary_report(statuses):
    """Generate summary report of completed experiments"""
    print("\nExperiment Results Summary:")
    print("-" * 70)

    results = []
    for exp_name in V4_EXPERIMENTS:
        metrics, status = statuses[exp_name]
        if status == "complete" and metrics:
            fmt_metrics = format_metrics(metrics)
            label = exp_name.replace("pb_ood_aug_v4_", "").upper()
            results.append({
                'name': label,
                'test_l2': fmt_metrics['test_l2'],
                'test_pde': fmt_metrics['test_pde'],
                'gap_pct': fmt_metrics['gap_pct'],
            })

    if results:
        # Find best result
        best_idx = min(range(len(results)),
                      key=lambda i: results[i]['test_l2']
                      if isinstance(results[i]['test_l2'], (int, float)) else float('inf'))

        for i, r in enumerate(results):
            marker = "  [BEST]" if i == best_idx else "         "
            tl2_str = "{:.4f}".format(r['test_l2']) if isinstance(r['test_l2'], (int, float)) else str(r['test_l2'])
            print("{} {}: Test_L2={} | Gap={:+.1f}%".format(
                marker, r['name'].ljust(15),
                tl2_str,
                r['gap_pct']
            ))

    print("\nExperiment outputs saved to:")
    for exp_name in V4_EXPERIMENTS:
        exp_path = OUTPUT_BASE / exp_name
        if exp_path.exists():
            print("  - {}".format(exp_path))

if __name__ == '__main__':
    monitor_v4_experiments()
