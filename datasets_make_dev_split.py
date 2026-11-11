"""从 train_000 中随机切出独立 dev 集，test 文件保持不变。

输出（原文件不动）:
  burgers_nu0.01_train_split.parquet
  burgers_nu0.01_dev_split.parquet
同时检查 train/dev/test 之间是否存在重复样本。
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SPLIT_SEED = 0          # 划分用的固定种子，与训练 seed 无关
DEV_FRACTION_OF_TOTAL = 0.15


def row_hashes(table):
    col = table.column("tensor")
    return [hashlib.sha1(np.asarray(col[i].as_py(), dtype=np.float32).tobytes()).hexdigest()
            for i in range(table.num_rows)]


def split(data_dir: Path):
    train = pq.read_table(data_dir / "burgers_nu0.01_train_000.parquet")
    test = pq.read_table(data_dir / "burgers_nu0.01_test.parquet")
    total = train.num_rows + test.num_rows
    n_dev = round(total * DEV_FRACTION_OF_TOTAL)

    perm = np.random.default_rng(SPLIT_SEED).permutation(train.num_rows)
    dev_idx, tr_idx = np.sort(perm[:n_dev]), np.sort(perm[n_dev:])
    tr, dev = train.take(tr_idx), train.take(dev_idx)

    h_tr, h_dev, h_te = set(row_hashes(tr)), set(row_hashes(dev)), set(row_hashes(test))
    overlaps = {"train∩dev": len(h_tr & h_dev), "train∩test": len(h_tr & h_te), "dev∩test": len(h_dev & h_te)}
    dup_in_train = tr.num_rows - len(h_tr)

    pq.write_table(tr, data_dir / "burgers_nu0.01_train_split.parquet")
    pq.write_table(dev, data_dir / "burgers_nu0.01_dev_split.parquet")

    n = (tr.num_rows, dev.num_rows, test.num_rows)
    print(f"{data_dir.name}: train/dev/test = {n} "
          f"= {n[0]/total:.1%}/{n[1]/total:.1%}/{n[2]/total:.1%}")
    print(f"  重复检查: {overlaps}, train 内部重复 {dup_in_train}")
    return all(v == 0 for v in overlaps.values())


ok = True
for name in ["burgers_custom_pb_id", "burgers_custom_pb_ood"]:
    ok &= split(Path("datasets") / name)
sys.exit(0 if ok else 1)
