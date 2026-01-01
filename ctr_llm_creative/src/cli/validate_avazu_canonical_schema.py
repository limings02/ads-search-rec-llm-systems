# src/cli/validate_avazu_canonical_schema.py
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.data.splits import SplitSpec
from src.data.adapters.avazu import AvazuAdapter


def assert_cols_exist(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"[{where}] missing columns: {missing}")


def dtype_name(s: pd.Series) -> str:
    # pandas StringDtype -> 'string'
    # object -> 'object'
    return str(s.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--max_batches_per_split", type=int, default=5, help="each split sample how many batches")
    args = ap.parse_args()

    # 你的切分：前 8 天 train，后 2 天 valid，test 外置
    split = SplitSpec(
        method="time",
        train_end_day=20141028,
        valid_days=[20141029, 20141030],
        test_day=None,
        test_days=None,
        assert_disjoint=True,
    )

    adp = AvazuAdapter(train_path=args.train_path, test_path=args.test_path, split=split)

    base_cols = ["_id", "_day", "_hod", "_label_click"]
    feat_cols = adp.get_features()

    # 统计信息
    seen_days: Dict[str, Set[int]] = defaultdict(set)
    missing_rate_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    dtype_stats: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    batches_seen = defaultdict(int)

    for split_name, df in adp.iter_splits(chunksize=args.chunksize):
        if batches_seen[split_name] >= args.max_batches_per_split:
            continue

        where = f"{split_name}/batch={batches_seen[split_name]}"

        # 1) 必备列检查
        assert_cols_exist(df, base_cols, where=where)

        # 2) 特征列必须完整
        assert_cols_exist(df, feat_cols, where=where)

        # 3) dtype 检查：特征列必须是 string（object 也算“危险”，直接报）
        bad = []
        for c in feat_cols:
            dt = dtype_name(df[c])
            dtype_stats[split_name][c].add(dt)
            if dt != "string":
                bad.append((c, dt))
        if bad:
            # 给出前几个错误，避免刷屏
            bad_preview = ", ".join([f"{c}:{dt}" for c, dt in bad[:10]])
            raise AssertionError(
                f"[{where}] feature dtype not 'string' for {len(bad)} cols. examples: {bad_preview}\n"
                f"Tip: canonicalize() should cast feature cols to pandas 'string' dtype."
            )

        # 4) 缺失率统计（__MISSING__）
        for c in feat_cols:
            miss_rate = (df[c] == "__MISSING__").mean()
            missing_rate_stats[split_name][c].append(float(miss_rate))

        # 5) day 覆盖统计
        for d in df["_day"].dropna().unique().tolist():
            seen_days[split_name].add(int(d))

        # 6) label 检查：train/valid 必须是 0/1；test 允许全 NA
        if split_name in ("train", "valid"):
            # 允许极少量空值？不建议。这里严格一点
            if df["_label_click"].isna().any():
                raise AssertionError(f"[{where}] {split_name} has NA labels, should not happen.")
            uniq = set(df["_label_click"].unique().tolist())
            if not uniq.issubset({0, 1}):
                raise AssertionError(f"[{where}] {split_name} label values abnormal: {sorted(list(uniq))[:10]}")
        else:
            # test：允许全 NA，但如果部分有 label 也行
            pass

        batches_seen[split_name] += 1

        # 如果三个 split 都够了就停
        if all(batches_seen[s] >= args.max_batches_per_split for s in ("train", "valid", "test")):
            break

    # 汇总输出
    print("=" * 80)
    print("CANONICAL SCHEMA VALIDATION: PASS")
    print("=" * 80)
    print("Feature cols:", len(feat_cols))
    print("Batches seen:", dict(batches_seen))

    print("\nDays seen per split:")
    for s in ("train", "valid", "test"):
        days = sorted(list(seen_days.get(s, set())))
        print(f"  {s}: {days[:10]}{'...' if len(days) > 10 else ''} (count={len(days)})")

    print("\nMissing-rate (avg over sampled batches): show top 10 highest missing columns per split")
    for s in ("train", "valid", "test"):
        col_avgs = []
        for c, rates in missing_rate_stats.get(s, {}).items():
            if rates:
                col_avgs.append((c, sum(rates) / len(rates)))
        col_avgs.sort(key=lambda x: x[1], reverse=True)
        print(f"  [{s}]")
        for c, r in col_avgs[:10]:
            print(f"    {c}: {r:.4f}")

    print("\nDtype summary (should all be 'string'):")
    for s in ("train", "valid", "test"):
        dts = set()
        for c in feat_cols:
            dts |= dtype_stats[s].get(c, set())
        print(f"  {s}: {sorted(list(dts))}")


if __name__ == "__main__":
    main()
