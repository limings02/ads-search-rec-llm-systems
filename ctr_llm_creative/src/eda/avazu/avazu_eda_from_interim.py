# src/eda/avazu/avazu_eda_from_interim.py
from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


# -----------------------------
# 轻量 HyperLogLog：估计“unique 数量”
# 用于：OOV unique 太大，不能用 set() 硬记
# -----------------------------
class HyperLogLog32:
    """
    一个足够用的 HLL(32-bit hash) 实现，估计 distinct。
    - p=14 => m=16384 registers，内存 ~16KB/实例
    - 用 crc32 做稳定 hash（比 md5 快很多）
    线上版本建议换 xxhash / murmurhash3（更快）。
    """
    def __init__(self, p: int = 14):
        assert 4 <= p <= 16
        self.p = p
        self.m = 1 << p
        self.reg = bytearray(self.m)

        # alpha_m 常数
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    @staticmethod
    def _hash32(x: str) -> int:
        # 稳定、跨进程一致
        return zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF

    @staticmethod
    def _clz32(x: int) -> int:
        # count leading zeros for 32-bit int
        if x == 0:
            return 32
        return 32 - x.bit_length()

    def add(self, x: str) -> None:
        h = self._hash32(x)
        idx = h & (self.m - 1)          # 低 p 位作为桶号
        w = h >> self.p                 # 剩余 bits
        # rank = leading_zeros(w) + 1（在剩余位宽内）
        rank = self._clz32(w) + 1
        if rank > self.reg[idx]:
            self.reg[idx] = rank

    def estimate(self) -> float:
        # E = alpha*m^2 / sum(2^-reg[i])
        inv_sum = 0.0
        zeros = 0
        for r in self.reg:
            inv_sum += 2.0 ** (-r)
            if r == 0:
                zeros += 1

        E = self.alpha * (self.m ** 2) / inv_sum

        # small-range correction: linear counting
        if E <= 2.5 * self.m and zeros > 0:
            # zeros 是空桶数，理论上 > 0，这里仍然做保护
            z = max(zeros, 1)          # 防御性：避免意外 0
            E = self.m * np.log(self.m / z)

        return float(E)


@dataclass
class AvazuEdaConfig:
    """
    EDA 参数（只读 interim/canonical）
    """
    canonical_root: str            # data/interim/avazu/canonical
    out_root: str                  # data/interim/avazu/eda
    batch_size: int = 500_000
    # 在 AvazuEdaConfig 里加两个字段
    coverage_ks: list[int] = None     # 你想落盘的 coverage 曲线点
    coverage_max_k: int = 5000        # 最多算到多少个 top 值（受 keep_for_count 上限影响）

    # TopK 相关
    topk: int = 50                 # 输出 top 50 值
    keep_for_count: int = 5000     # 内存保护：每个特征全局最多保留多少个候选计数（近似 heavy hitters）
    per_batch_topm: int = 2000     # 每个 batch 只拿 topM 更新（减少长尾污染）
    vocab_k: int = 2000            # 用 train 的 top vocab_k 作为 “vocab 候选”，用于算 OOV（valid/test）

    # 列名约定（和你 canonical 一致）
    day_col: str = "_day"
    hod_col: str = "_hod"
    label_col: str = "_label_click"

    # 缺失占位符（你 adapter 里用的 __MISSING__）
    missing_token: str = "__MISSING__"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _open_split_dataset(canonical_root: str, split_name: str):
    """
    关键修复：不要对 canonical_root 总目录建 dataset。
    改为对 split 子目录建 dataset，避免 _label_click 在 train(int8)/test(null) 间做 schema 统一导致 cast 崩溃。
    """
    import pyarrow.dataset as ds

    split_root = os.path.join(canonical_root, f"split={split_name}")
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    dataset = ds.dataset(split_root, format="parquet", partitioning="hive")
    return ds, dataset


def _iter_batches(canonical_root: str, split_name: str, columns: List[str], batch_size: int) -> Iterator[pd.DataFrame]:
    """
    流式读取指定 split（train/valid/test），从 split 子目录构建 dataset。
    """
    ds, dataset = _open_split_dataset(canonical_root, split_name)

    # split 子目录下不需要再 filter split
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for rb in scanner.to_batches():
        yield rb.to_pandas()



def _bounded_update_counter(global_counter: Counter, batch_series: pd.Series,
                            per_batch_topm: int, keep_for_count: int) -> None:
    """
    内存安全地累计频次：
    - 只取 batch 的 topM 来更新
    - 更新后把全局 counter 裁剪到 keep_for_count
    这是 heavy hitters 的近似实现（足够做工程决策）。
    """
    vc = batch_series.value_counts(dropna=False).head(per_batch_topm)
    global_counter.update(vc.to_dict())

    # 裁剪：只保留最大的 keep_for_count 个 key
    if len(global_counter) > keep_for_count:
        top_items = global_counter.most_common(keep_for_count)
        global_counter.clear()
        global_counter.update(dict(top_items))


def run_avazu_eda(cfg: AvazuEdaConfig, feature_cols: List[str]) -> None:
    """
    主入口：读 interim/canonical -> 产出 EDA 文件到 out_root
    """
    _ensure_dir(cfg.out_root)
    assert cfg.vocab_k <= cfg.keep_for_count, "vocab_k must be <= keep_for_count to be meaningful"
    # -------------------------
    # 1) CTR by day / hod（train+valid；test 可能无 label）
    # -------------------------
    def scan_ctr(split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = [cfg.day_col, cfg.hod_col, cfg.label_col]
        day_imp = Counter()
        day_clk = Counter()
        hod_imp = Counter()
        hod_clk = Counter()

        for df in _iter_batches(cfg.canonical_root, split_name=split_name, columns=cols, batch_size=cfg.batch_size):
            # label 可能是 NA（test），这里仅统计非 NA
            y = pd.to_numeric(df[cfg.label_col], errors="coerce")
            m = y.notna()
            if m.sum() == 0:
                continue
            d = df.loc[m, cfg.day_col].astype("int64")
            h = df.loc[m, cfg.hod_col].astype("int64")
            yv = y.loc[m].astype("int64")

            # day
            day_imp.update(d.value_counts().to_dict())
            day_clk.update(yv.groupby(d).sum().to_dict())
            # hod
            hod_imp.update(h.value_counts().to_dict())
            hod_clk.update(yv.groupby(h).sum().to_dict())

        # 汇总
        day_rows = []
        for day in sorted(day_imp.keys()):
            imp = int(day_imp[day])
            clk = int(day_clk.get(day, 0))
            ctr = clk / imp if imp > 0 else 0.0
            day_rows.append({"split": split_name, "day": int(day), "impressions": imp, "clicks": clk, "ctr": ctr})
        hod_rows = []
        for hod in sorted(hod_imp.keys()):
            imp = int(hod_imp[hod])
            clk = int(hod_clk.get(hod, 0))
            ctr = clk / imp if imp > 0 else 0.0
            hod_rows.append({"split": split_name, "hod": int(hod), "impressions": imp, "clicks": clk, "ctr": ctr})

        return pd.DataFrame(day_rows), pd.DataFrame(hod_rows)

    day_train, hod_train = scan_ctr("train")
    day_valid, hod_valid = scan_ctr("valid")
    # test 若无 label，会输出空表
    day_test, hod_test = scan_ctr("test")

    pd.concat([day_train, day_valid, day_test], ignore_index=True).to_parquet(
        os.path.join(cfg.out_root, "ctr_by_day.parquet"), index=False
    )
    pd.concat([hod_train, hod_valid, hod_test], ignore_index=True).to_parquet(
        os.path.join(cfg.out_root, "ctr_by_hod.parquet"), index=False
    )

    # -------------------------
    # 2) train-only TopK 频次 + 覆盖率（用于 FeatureMap 决策）
    # -------------------------
    total_rows_train = 0
    missing_counts = Counter()  # feature -> missing rows
    counters: Dict[str, Counter] = {f: Counter() for f in feature_cols}

    cols = feature_cols  # train-only
    for df in _iter_batches(cfg.canonical_root, split_name="train", columns=cols, batch_size=cfg.batch_size):
        total_rows_train += len(df)
        for f in feature_cols:
            s = df[f].astype("string")
            missing_counts[f] += int((s == cfg.missing_token).sum())
            _bounded_update_counter(
                counters[f], s, per_batch_topm=cfg.per_batch_topm, keep_for_count=cfg.keep_for_count
            )


    # -------------------------
    # 输出 topk 表 + 覆盖率曲线 + min_k(targets)
    # -------------------------
    topk_rows = []
    coverage_rows = []
    missing_rows = []
    min_k_rows = []

    # 1) 默认的 coverage 采样点：工程上常用“对数/分段网格”
    if cfg.coverage_ks is None:
        cfg.coverage_ks = [1,2,3,5,10,20,50,100,200,500,1000,1500,2000,3000,5000]

    coverage_ks_set = set(cfg.coverage_ks + [cfg.vocab_k])  # 确保 vocab_k 一定有
    targets = [(0.90, "min_k@90"), (0.95, "min_k@95"), (0.99, "min_k@99")]

    for f in feature_cols:
        denom = max(total_rows_train - int(missing_counts[f]), 1)
        missing_rows.append({
            "feature": f,
            "missing_rate_train": float(missing_counts[f] / max(total_rows_train, 1))
        })

        # 关键：不要只 most_common(topk)，而是 most_common(max_k)
        # 注意：max_k 不要超过 keep_for_count，否则也没有意义（因为 counter 已被裁剪）
        max_k = min(cfg.coverage_max_k, cfg.keep_for_count)

        items = counters[f].most_common(max_k)

        cum = 0
        # 记录 min_k 的状态
        min_k = {name: None for _, name in targets}

        for rank, (val, cnt) in enumerate(items, start=1):
            cnt = int(cnt)
            cum += cnt
            cov = cum / denom

            # (a) topk 明细仍然只输出 cfg.topk 行（保持文件小）
            if rank <= cfg.topk:
                topk_rows.append({
                    "feature": f,
                    "rank": rank,
                    "value": str(val),
                    "count": cnt,
                    "pct_of_train_rows": cnt / max(total_rows_train, 1),
                })

            # (b) coverage 曲线：只在指定 ks 上落点（大幅减少文件体积）
            if rank in coverage_ks_set:
                coverage_rows.append({
                    "feature": f,
                    "k": rank,
                    "coverage_non_missing": float(cov),
                })

            # (c) min_k：第一次达到阈值就记录
            for thr, name in targets:
                if min_k[name] is None and cov >= thr:
                    min_k[name] = rank

        # (d) 如果 max_k 内达不到阈值，用一个可解释的标记（别留空）
        # (d) 输出 min_k：保持数值类型 + reached 标记（避免 Parquet 类型冲突）
        out = {
            "feature": f,
            "max_k_used": int(max_k),         # 关键：用于解释 “>max_k”
            "coverage@vocab_k": None,
        }

        # coverage@vocab_k：如果 vocab_k > max_k，则不可得（因为你只统计到 max_k）
        if cfg.vocab_k <= max_k:
            cov_vocab = sum(int(c) for _, c in items[:cfg.vocab_k]) / denom
            out["coverage@vocab_k"] = float(cov_vocab)
        else:
            out["coverage@vocab_k"] = np.nan  # 用 NaN/NA 表示不可得

        for _, name in targets:
            reached = (min_k[name] is not None)
            out[name] = int(min_k[name]) if reached else None          # 数值 or None
            out[f"{name}_reached"] = bool(reached)                     # reached 标记

        min_k_rows.append(out)


    # 落盘
    pd.DataFrame(topk_rows).to_parquet(os.path.join(cfg.out_root, "topk_train.parquet"), index=False)
    pd.DataFrame(coverage_rows).to_parquet(os.path.join(cfg.out_root, "coverage_train.parquet"), index=False)
    pd.DataFrame(missing_rows).to_parquet(os.path.join(cfg.out_root, "missing_train.parquet"), index=False)
    df_min_k = pd.DataFrame(min_k_rows)

    # 强制三列为可空整数（关键：pandas 的 Int64 不是 numpy int64）
    for col in ["min_k@90", "min_k@95", "min_k@99"]:
        df_min_k[col] = pd.to_numeric(df_min_k[col], errors="coerce").astype("Int64")

    # reached 列为 bool（可选，但建议）
    for col in ["min_k@90_reached", "min_k@95_reached", "min_k@99_reached"]:
        df_min_k[col] = df_min_k[col].astype("bool")

    df_min_k.to_parquet(os.path.join(cfg.out_root, "min_k_train.parquet"), index=False)



    # -------------------------
    # 3) OOV 统计：valid/test 相对 train vocab（vocab_k）
    # -------------------------
    # 构建 train vocab set（每个特征一个 set，大小 vocab_k，完全可控）
    vocab: Dict[str, set] = {}
    for f in feature_cols:
        vocab[f] = set(val for val, _ in counters[f].most_common(cfg.vocab_k))

    def scan_oov(split_name: str) -> pd.DataFrame:
        cols = [cfg.day_col] + feature_cols
        total = Counter()      # feature -> rows
        oov_rows = Counter()   # feature -> oov rows
        oov_hll: Dict[str, HyperLogLog32] = {f: HyperLogLog32(p=14) for f in feature_cols}

        for df in _iter_batches(cfg.canonical_root, split_name=split_name, columns=cols, batch_size=cfg.batch_size):
            for f in feature_cols:
                s = df[f].astype("string")
                total[f] += len(s)

                # missing 不计入 oov（你也可以选择算作 oov，这里更符合工程：missing 单独处理）
                m_valid = (s != cfg.missing_token)
                if m_valid.sum() == 0:
                    continue

                sv = s[m_valid]
                # oov：不在 train vocab
                m_oov = ~sv.isin(vocab[f])
                oov_rows[f] += int(m_oov.sum())

                # OOV unique：用 HLL 近似（避免 set 爆内存）
                # 只对 OOV 值 update
                for v in sv[m_oov].tolist():
                    oov_hll[f].add(str(v))

        rows = []
        for f in feature_cols:
            tot = int(total[f])
            oov = int(oov_rows[f])
            rate = oov / tot if tot > 0 else 0.0
            uniq = oov_hll[f].estimate() if oov > 0 else 0.0
            rows.append({
                "split": split_name,
                "feature": f,
                "vocab_k": cfg.vocab_k,
                "oov_row_rate": float(rate),
                "oov_unique_est": float(uniq),
            })
        return pd.DataFrame(rows)

    oov_valid = scan_oov("valid")
    oov_test = scan_oov("test")
    pd.concat([oov_valid, oov_test], ignore_index=True).to_parquet(
        os.path.join(cfg.out_root, "oov_vs_train_vocab.parquet"), index=False
    )

    # 写一个简单 summary（方便你肉眼快速 check）
    with open(os.path.join(cfg.out_root, "README_eda_outputs.txt"), "w", encoding="utf-8") as f:
        f.write("EDA outputs:\n")
        f.write("- ctr_by_day.parquet\n")
        f.write("- ctr_by_hod.parquet\n")
        f.write("- topk_train.parquet\n")
        f.write("- coverage_train.parquet\n")
        f.write("- missing_train.parquet\n")
        f.write("- oov_vs_train_vocab.parquet\n")
