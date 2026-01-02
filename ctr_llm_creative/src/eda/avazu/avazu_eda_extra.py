# src/eda/avazu/avazu_eda_extra.py
from __future__ import annotations

import os
import math
import zlib
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.utils.path_resolver import resolve_paths_in_config

# =========================================================
# 读数据：只从 interim/canonical 读，且按 split 子目录建 dataset
# （避免 train(label=int8) 与 test(label=null) 的 Arrow schema 冲突）
# =========================================================
def _open_split_dataset(canonical_root: str, split: str):
    import pyarrow.dataset as ds
    split_root = os.path.join(canonical_root, f"split={split}")
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    dataset = ds.dataset(split_root, format="parquet", partitioning="hive")
    return dataset


def iter_split_batches(
    canonical_root: str,
    split: str,
    columns: List[str],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """
    流式读 split 子目录，避免一次性读入全量数据。
    """
    dataset = _open_split_dataset(canonical_root, split)
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for rb in scanner.to_batches():
        yield rb.to_pandas()


# =========================================================
# 近似数据结构：Bloom（用于“是否在 train 出现过” membership test）
# - false positive 会“低估新值率”，但对 EDA 足够
# =========================================================
class BloomFilter:
    """
    轻量 BloomFilter：bytearray bitset + 两个 hash
    - m_bits: bitset 大小（位）
    - k=2: 两个 hash（足够快）
    """
    def __init__(self, m_bits: int = 1 << 26):
        # 2^26 bits ~= 67M bits ~= 8MB
        self.m_bits = int(m_bits)
        self.m_mask = self.m_bits - 1
        if self.m_bits & self.m_mask != 0:
            raise ValueError("m_bits must be power of 2 for fast mask.")
        self.bits = bytearray(self.m_bits // 8)

    @staticmethod
    def _h1(x: str) -> int:
        return zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF

    @staticmethod
    def _h2(x: str) -> int:
        # adler32 更快一些；作为第二个 hash 足够
        return zlib.adler32(x.encode("utf-8")) & 0xFFFFFFFF

    def _setbit(self, idx: int) -> None:
        byte_i = idx >> 3
        bit_i = idx & 7
        self.bits[byte_i] |= (1 << bit_i)

    def _getbit(self, idx: int) -> int:
        byte_i = idx >> 3
        bit_i = idx & 7
        return (self.bits[byte_i] >> bit_i) & 1

    def add(self, x: str) -> None:
        i1 = self._h1(x) & self.m_mask
        i2 = self._h2(x) & self.m_mask
        self._setbit(i1)
        self._setbit(i2)

    def __contains__(self, x: str) -> bool:
        i1 = self._h1(x) & self.m_mask
        i2 = self._h2(x) & self.m_mask
        return (self._getbit(i1) == 1) and (self._getbit(i2) == 1)


# =========================================================
# 统计学工具：PSI / Wilson CI / z-test
# =========================================================
def psi_from_distributions(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """
    PSI = sum( (q-p)*ln(q/p) )
    - p: baseline（通常 train）
    - q: target（valid/test 或某天）
    注意：p/q 可能出现 0，需要 eps 平滑。
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum((q - p) * np.log(q / p)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for Bernoulli proportion.
    - k: successes（clicks）
    - n: trials（impressions）
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def zscore_vs_global(k: int, n: int, p0: float) -> float:
    """
    一样本比例 z-test：检验某 value 的 CTR 是否显著偏离全局 CTR
    z = (p_hat - p0) / sqrt(p0*(1-p0)/n)
    """
    if n <= 0:
        return 0.0
    phat = k / n
    denom = math.sqrt(max(p0 * (1.0 - p0) / n, 1e-12))
    return float((phat - p0) / denom)


# =========================================================
# 配置
# =========================================================
@dataclass
class AvazuEdaExtraConfig:
    canonical_root: str               # data/interim/avazu/canonical
    out_root: str                     # data/interim/avazu/eda_extra
    batch_size: int = 500_000

    # 列名（与你 canonical 一致）
    day_col: str = "_day"
    hod_col: str = "_hod"
    label_col: str = "_label_click"

    # PSI
    psi_topk_bins: int = 50           # 每个特征取 train topK 作为 bins，其余为 __OTHER__
    psi_other_token: str = "__OTHER__"

    # 新值率（Bloom）
    bloom_bits: int = 1 << 26         # 每个特征 Bloom bitset 大小（默认 8MB）
    # CTR by value
    ctr_value_topn: int = 2000        # 每个特征只对 train topN value 做统计（更稳、更省内存）
    ctr_min_imps_list: Tuple[int, ...] = (10_000, 50_000)  # 输出两个阈值视角
    # 交互
    pair_topk_each: int = 30          # 每个特征取 topK 值，组合最多 K^2（建议 20~50）
    pair_min_imps: int = 20_000       # 过滤低曝光组合，避免噪声
    # 缺失 token（你目前 missing_rate=0，但代码留着，泛化到其它数据集）
    missing_token: str = "__MISSING__"


# =========================================================
# 1) PSI：TopK bins + OTHER，按 day 输出
# =========================================================
def build_train_top_bins(
    cfg: AvazuEdaExtraConfig,
    feature_cols: List[str],
) -> Dict[str, set]:
    """
    构建 PSI bins：每个特征取 train 的 topK value 作为“单独 bin”，其余进 OTHER。
    - 这里只做频次统计（不需要 label）
    - 用 Counter 做 topK 即可
    """
    top_bins: Dict[str, Counter] = {f: Counter() for f in feature_cols}
    cols = [cfg.day_col] + feature_cols

    for df in iter_split_batches(cfg.canonical_root, split="train", columns=cols, batch_size=cfg.batch_size):
        for f in feature_cols:
            s = df[f].astype("string")
            # 这里不需要保留全量，只要 topK；先更新，再裁剪（避免 Counter 无限膨胀）
            vc = s.value_counts(dropna=False).head(max(cfg.psi_topk_bins * 5, 500))
            top_bins[f].update(vc.to_dict())
            # 裁剪：只保留 topK*5 作为候选（下一批继续竞争）
            if len(top_bins[f]) > cfg.psi_topk_bins * 5:
                top_bins[f] = Counter(dict(top_bins[f].most_common(cfg.psi_topk_bins * 5)))

    # 最终取 topK
    bins = {f: set([v for v, _ in top_bins[f].most_common(cfg.psi_topk_bins)]) for f in feature_cols}
    return bins


def run_psi_by_field_day(
    cfg: AvazuEdaExtraConfig,
    feature_cols: List[str],
    top_bins: Dict[str, set],
    splits: Tuple[str, ...] = ("train", "valid", "test"),
) -> None:
    """
    输出：
      - psi_by_field_day.parquet：feature/split/day/psi
      - psi_components_day.parquet：可选（每个 bin 的贡献）
    说明：
      baseline p：train 全量（聚合所有 train day）
      target q：每个 split 的每一天
    """
    cols = [cfg.day_col] + feature_cols

    # counts[(split, day, feature, bin)] = cnt
    counts = defaultdict(int)
    # totals[(split, day, feature)] = rows
    totals = defaultdict(int)

    for sp in splits:
        for df in iter_split_batches(cfg.canonical_root, split=sp, columns=cols, batch_size=cfg.batch_size):
            day = df[cfg.day_col].astype("int64")
            for f in feature_cols:
                s = df[f].astype("string")

                # binning：topK 进独立 bin，其余进 OTHER
                bset = top_bins[f]
                b = s.where(s.isin(bset), other=cfg.psi_other_token)

                # missing 单独留出来（更合理；否则 missing 会混进 OTHER）
                b = b.where(b != cfg.missing_token, other=cfg.missing_token)

                # 统计 (day, bin) 频次
                # MultiIndex value_counts 比 crosstab 更省内存
                tmp = pd.DataFrame({"day": day, "bin": b})
                vc = tmp.value_counts().reset_index(name="cnt")  # columns: day/bin/cnt
                for _, r in vc.iterrows():
                    key = (sp, int(r["day"]), f, str(r["bin"]))
                    counts[key] += int(r["cnt"])
                    totals[(sp, int(r["day"]), f)] += int(r["cnt"])

    # baseline：train 全量（聚合所有 train day）
    baseline = defaultdict(int)   # baseline[(feature, bin)] = cnt
    baseline_total = defaultdict(int)
    for (sp, day, f, b), c in counts.items():
        if sp != "train":
            continue
        baseline[(f, b)] += c
        baseline_total[f] += c

    # 计算 PSI
    rows = []
    comp_rows = []
    for (sp, day, f), tot in totals.items():
        # 构造 bins 列表：topK bins + OTHER + MISSING（如果存在）
        bins_list = list(top_bins[f]) + [cfg.psi_other_token, cfg.missing_token]
        # baseline p
        p = np.array([baseline.get((f, b), 0) for b in bins_list], dtype=np.float64)
        p = p / max(float(baseline_total.get(f, 0)), 1.0)
        # target q
        q = np.array([counts.get((sp, day, f, b), 0) for b in bins_list], dtype=np.float64)
        q = q / max(float(tot), 1.0)

        psi = psi_from_distributions(p, q)
        rows.append({"feature": f, "split": sp, "day": int(day), "psi": float(psi), "n_rows": int(tot)})

        # component（可解释：哪个 bin 贡献最大）
        eps = 1e-10
        pp = np.clip(p, eps, 1.0)
        qq = np.clip(q, eps, 1.0)
        contrib = (qq - pp) * np.log(qq / pp)
        for b, cc in zip(bins_list, contrib):
            comp_rows.append({
                "feature": f, "split": sp, "day": int(day), "bin": str(b),
                "p": float(pp[bins_list.index(b)]), "q": float(qq[bins_list.index(b)]),
                "psi_contrib": float(cc),
            })

    out = pd.DataFrame(rows).sort_values(["feature", "split", "day"])
    out.to_parquet(os.path.join(cfg.out_root, "psi_by_field_day.parquet"), index=False)

    comp = pd.DataFrame(comp_rows)
    comp.to_parquet(os.path.join(cfg.out_root, "psi_components_day.parquet"), index=False)


# =========================================================
# 2) 新值率：Bloom
# =========================================================
def run_new_value_rate(
    cfg: AvazuEdaExtraConfig,
    feature_cols: List[str],
    top_bins: Dict[str, set],
) -> None:
    """
    输出：new_value_rate_by_day.parquet

    两种口径：
    A) train 内按天：用“增量 Bloom”，统计当天新值行占比（更能体现 drift/novelty）
    B) valid/test：相对“全量 train Bloom”，统计新值行占比（近似）
    """
    cols = [cfg.day_col] + feature_cols

    # ---- A) train incremental ----
    # 为每个 feature 一个 bloom（内存：feature_num * bloom_bits/8）
    blooms_train = {f: BloomFilter(cfg.bloom_bits) for f in feature_cols}

    # 先把 train 的天列出并排序（避免一次性读全量，仅扫描 day）
    train_days = set()
    for df in iter_split_batches(cfg.canonical_root, split="train", columns=[cfg.day_col], batch_size=cfg.batch_size):
        train_days.update(df[cfg.day_col].astype("int64").unique().tolist())
    train_days = sorted(int(x) for x in train_days)

    rows = []

    # 逐天扫描 train（注意：dataset 按 hive partition 是 split+day 的目录，读全 split 再过滤 day 会浪费）
    # 这里简单实现：读 train 全量 batch 再按 day 分组更新；工程上更快的方法是直接遍历目录 day=xxxx。
    for day in train_days:
        day_total = Counter()
        day_new = Counter()

        for df in iter_split_batches(cfg.canonical_root, split="train", columns=cols, batch_size=cfg.batch_size):
            d = df[cfg.day_col].astype("int64")
            m = (d == day)
            if int(m.sum()) == 0:
                continue
            sub = df.loc[m, feature_cols]

            for f in feature_cols:
                s = sub[f].astype("string")
                day_total[f] += len(s)

                # 逐个 membership check（Python 循环会慢，但可接受；你后面可以只对“高漂移字段”跑）
                new_cnt = 0
                bf = blooms_train[f]
                for v in s.tolist():
                    v = str(v)
                    if v == cfg.missing_token:
                        continue
                    if v not in bf:
                        new_cnt += 1
                        bf.add(v)  # 注意：增量更新
                    else:
                        # 已见过也要 add（无害）
                        bf.add(v)
                day_new[f] += new_cnt

        for f in feature_cols:
            tot = int(day_total[f])
            nw = int(day_new[f])
            rows.append({
                "scope": "train_incremental",
                "split": "train",
                "day": int(day),
                "feature": f,
                "new_row_rate": float(nw / tot) if tot > 0 else 0.0,
                "n_rows": tot,
            })

    # ---- B) valid/test vs full-train bloom ----
    # 重建“全量 train bloom”（避免被 A 的增量状态影响）
    blooms_full = {f: BloomFilter(cfg.bloom_bits) for f in feature_cols}
    for df in iter_split_batches(cfg.canonical_root, split="train", columns=feature_cols, batch_size=cfg.batch_size):
        for f in feature_cols:
            s = df[f].astype("string")
            bf = blooms_full[f]
            for v in s.tolist():
                v = str(v)
                if v == cfg.missing_token:
                    continue
                bf.add(v)

    for sp in ("valid", "test"):
        day_total = Counter()
        day_new = Counter()

        for df in iter_split_batches(cfg.canonical_root, split=sp, columns=cols, batch_size=cfg.batch_size):
            d = df[cfg.day_col].astype("int64")
            for day, sub in df.groupby(d, sort=False):
                for f in feature_cols:
                    s = sub[f].astype("string")
                    day_total[(int(day), f)] += len(s)
                    bf = blooms_full[f]
                    new_cnt = 0
                    for v in s.tolist():
                        v = str(v)
                        if v == cfg.missing_token:
                            continue
                        if v not in bf:
                            new_cnt += 1
                    day_new[(int(day), f)] += new_cnt

        for (day, f), tot in day_total.items():
            nw = int(day_new[(day, f)])
            rows.append({
                "scope": "vs_full_train",
                "split": sp,
                "day": int(day),
                "feature": f,
                "new_row_rate": float(nw / tot) if tot > 0 else 0.0,
                "n_rows": int(tot),
            })

    out = pd.DataFrame(rows).sort_values(["scope", "split", "feature", "day"])
    out.to_parquet(os.path.join(cfg.out_root, "new_value_rate_by_day.parquet"), index=False)


# =========================================================
# 3) CTR by value：Wilson CI + zscore（train）
# =========================================================
def run_ctr_by_value_significance(
    cfg: AvazuEdaExtraConfig,
    feature_cols: List[str],
    top_bins: Dict[str, set],
) -> None:
    """
    目标：
    - 给每个特征的 topN value 统计 (imps, clicks, ctr, ci, zscore)
    - 用 min_imps 阈值避免被低样本误导
    输出：
      ctr_by_value_train.parquet
    """
    cols = [cfg.label_col] + feature_cols

    # 先用 train topN 候选：为了省内存，直接复用 PSI 的 top_bins 也可以；
    # 但 CTR by value 更希望 topN 大一些，所以这里用 Counter 再扩一次。
    cand: Dict[str, set] = {f: set() for f in feature_cols}
    freq_counter: Dict[str, Counter] = {f: Counter() for f in feature_cols}

    for df in iter_split_batches(cfg.canonical_root, split="train", columns=feature_cols, batch_size=cfg.batch_size):
        for f in feature_cols:
            s = df[f].astype("string")
            vc = s.value_counts(dropna=False).head(max(cfg.ctr_value_topn * 2, 2000))
            freq_counter[f].update(vc.to_dict())
            # 裁剪：只保留 2*topn 候选，防止 Counter 失控
            if len(freq_counter[f]) > cfg.ctr_value_topn * 2:
                freq_counter[f] = Counter(dict(freq_counter[f].most_common(cfg.ctr_value_topn * 2)))

    for f in feature_cols:
        cand[f] = set([v for v, _ in freq_counter[f].most_common(cfg.ctr_value_topn)])

    # 统计 clicks / imps
    imps = defaultdict(int)   # imps[(feature, value)] = n
    clks = defaultdict(int)   # clks[(feature, value)] = k
    global_imps = 0
    global_clks = 0

    for df in iter_split_batches(cfg.canonical_root, split="train", columns=cols, batch_size=cfg.batch_size):
        y = pd.to_numeric(df[cfg.label_col], errors="coerce").fillna(0).astype("int64")
        global_imps += len(y)
        global_clks += int(y.sum())

        for f in feature_cols:
            s = df[f].astype("string")
            # 只统计候选值；其它并到 __OTHER__（可选：你也可以不统计 OTHER）
            cset = cand[f]
            v = s.where(s.isin(cset), other=cfg.psi_other_token)
            # groupby value 统计 imps
            vc = v.value_counts(dropna=False)
            for val, n in vc.items():
                imps[(f, str(val))] += int(n)
            # groupby value 统计 clicks（把 label 聚合到 value 上）
            tmp = pd.DataFrame({"v": v, "y": y})
            ck = tmp.groupby("v")["y"].sum()
            for val, k in ck.items():
                clks[(f, str(val))] += int(k)

    p0 = global_clks / max(global_imps, 1)

    rows = []
    for f in feature_cols:
        # 只输出候选 + OTHER（避免输出无穷多行）
        values = list(cand[f]) + [cfg.psi_other_token]
        for val in values:
            n = int(imps.get((f, str(val)), 0))
            k = int(clks.get((f, str(val)), 0))
            if n <= 0:
                continue
            ctr = k / n
            lo, hi = wilson_ci(k, n)
            z = zscore_vs_global(k, n, p0=p0)
            rows.append({
                "feature": f,
                "value": str(val),
                "imps": n,
                "clicks": k,
                "ctr": float(ctr),
                "lift_vs_global": float(ctr / max(p0, 1e-12)),
                "wilson_lo": float(lo),
                "wilson_hi": float(hi),
                "zscore_vs_global": float(z),
            })

    out = pd.DataFrame(rows)
    # 给两个 min_imps 视角打标
    for thr in cfg.ctr_min_imps_list:
        out[f"pass_min_imps_{thr}"] = out["imps"] >= int(thr)

    # 一个可用的“疑似泄漏/异常”粗筛：高曝光且 zscore 很大
    out["is_suspicious"] = (out["imps"] >= int(cfg.ctr_min_imps_list[-1])) & (out["zscore_vs_global"].abs() >= 8.0)

    out = out.sort_values(["feature", "imps"], ascending=[True, False])
    out.to_parquet(os.path.join(cfg.out_root, "ctr_by_value_train.parquet"), index=False)


# =========================================================
# 4) Pairwise interaction：限定 topK bin，统计组合 lift 与 interaction score
# =========================================================
def run_pairwise_interaction(
    cfg: AvazuEdaExtraConfig,
    feature_cols: List[str],
    top_bins: Dict[str, set],
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    输出：
      - pair_interaction_summary.parquet：每个 pair 的 MI / top combo 等
      - pair_interaction_combos.parquet：每个 pair 的 top combos（组合级证据）

    注意：
      - 组合空间爆炸，所以必须“限定 topK 值 + OTHER”
      - 默认 pairs：挑几组广告业务常见强交互（你也可以从 CLI 传）
    """
    if pairs is None:
        pairs = [
            ("site_id", "app_id"),
            ("site_domain", "app_domain"),
            ("app_id", "device_model"),
            ("C14", "C17"),
        ]
        # 过滤掉不存在的
        pairs = [(a, b) for a, b in pairs if (a in feature_cols and b in feature_cols)]

    cols = [cfg.label_col, cfg.day_col] + list({x for p in pairs for x in p})

    # 全局 CTR
    global_imps = 0
    global_clks = 0

    # 组合统计：pair -> Counter((a_bin, b_bin) -> (imps, clicks))
    # 用 dict 存 tuple 值，避免嵌套 dict 太深
    combo_imps = defaultdict(int)
    combo_clks = defaultdict(int)
    a_imps = defaultdict(int)
    a_clks = defaultdict(int)
    b_imps = defaultdict(int)
    b_clks = defaultdict(int)

    for df in iter_split_batches(cfg.canonical_root, split="train", columns=cols, batch_size=cfg.batch_size):
        y = pd.to_numeric(df[cfg.label_col], errors="coerce").fillna(0).astype("int64")
        global_imps += len(y)
        global_clks += int(y.sum())

        for a, b in pairs:
            sa = df[a].astype("string")
            sb = df[b].astype("string")

            # binning：topK each + OTHER + MISSING
            a_top = set(list(top_bins.get(a, set()))[: cfg.pair_topk_each]) if isinstance(top_bins.get(a, set()), list) else top_bins.get(a, set())
            b_top = set(list(top_bins.get(b, set()))[: cfg.pair_topk_each]) if isinstance(top_bins.get(b, set()), list) else top_bins.get(b, set())

            # 更稳：这里不要直接切片 set；重新从 train 频次 top 获取更合理
            # 简化：复用 top_bins 并允许你调大 psi_topk_bins 以覆盖 pair_topk_each

            ba = sa.where(sa.isin(a_top), other=cfg.psi_other_token)
            bb = sb.where(sb.isin(b_top), other=cfg.psi_other_token)
            ba = ba.where(ba != cfg.missing_token, other=cfg.missing_token)
            bb = bb.where(bb != cfg.missing_token, other=cfg.missing_token)

            # 组合 key：pair|a_bin|b_bin
            key_pair = f"{a}__X__{b}"

            tmp = pd.DataFrame({"a": ba, "b": bb, "y": y})
            # imps by combo
            vc = tmp.groupby(["a", "b"]).size()
            ck = tmp.groupby(["a", "b"])["y"].sum()

            for (va, vb), n in vc.items():
                k = int(ck.loc[(va, vb)]) if (va, vb) in ck.index else 0
                combo_imps[(key_pair, str(va), str(vb))] += int(n)
                combo_clks[(key_pair, str(va), str(vb))] += int(k)

            # marginals
            vca = tmp.groupby("a").size()
            cka = tmp.groupby("a")["y"].sum()
            for va, n in vca.items():
                a_imps[(key_pair, str(va))] += int(n)
                a_clks[(key_pair, str(va))] += int(cka.loc[va])

            vcb = tmp.groupby("b").size()
            ckb = tmp.groupby("b")["y"].sum()
            for vb, n in vcb.items():
                b_imps[(key_pair, str(vb))] += int(n)
                b_clks[(key_pair, str(vb))] += int(ckb.loc[vb])

    p0 = global_clks / max(global_imps, 1)

    combo_rows = []
    summary_rows = []

    # 逐 pair 汇总
    for a, b in pairs:
        key_pair = f"{a}__X__{b}"

        # 取该 pair 所有 combo keys
        keys = [k for k in combo_imps.keys() if k[0] == key_pair]
        if not keys:
            continue

        # 计算 MI(Y;Xcombo)（X=组合bin）
        N = float(global_imps)
        p1 = float(p0)
        p0n = 1.0 - p1

        mi = 0.0
        best = None

        for (_, va, vb) in keys:
            n = int(combo_imps[(key_pair, va, vb)])
            k = int(combo_clks[(key_pair, va, vb)])
            if n <= 0:
                continue

            ctr = k / n
            lift_joint = ctr / max(p0, 1e-12)

            # 组合相对“边际乘积”的 interaction（近似）
            na = int(a_imps.get((key_pair, va), 0))
            ka = int(a_clks.get((key_pair, va), 0))
            nb = int(b_imps.get((key_pair, vb), 0))
            kb = int(b_clks.get((key_pair, vb), 0))
            ctr_a = (ka / na) if na > 0 else p0
            ctr_b = (kb / nb) if nb > 0 else p0

            expected = (ctr_a * ctr_b) / max(p0, 1e-12)  # 乘积模型的期望
            lift_over_marginal = ctr / max(expected, 1e-12)
            log_interaction = math.log(max(lift_over_marginal, 1e-12))

            # MI components
            # p(x,1), p(x,0), p(x)
            px = n / N
            px1 = k / N
            px0 = (n - k) / N
            if px1 > 0:
                mi += px1 * math.log(px1 / max(px * p1, 1e-12))
            if px0 > 0:
                mi += px0 * math.log(px0 / max(px * p0n, 1e-12))

            if (best is None) or (n >= cfg.pair_min_imps and abs(log_interaction) > abs(best["log_interaction"])):
                best = {
                    "a_bin": va, "b_bin": vb, "imps": n, "clicks": k,
                    "ctr": ctr, "lift_joint": lift_joint,
                    "lift_over_marginal": lift_over_marginal,
                    "log_interaction": log_interaction,
                }

            combo_rows.append({
                "pair": key_pair,
                "feature_a": a, "feature_b": b,
                "a_bin": va, "b_bin": vb,
                "imps": n, "clicks": k,
                "ctr": float(ctr),
                "lift_joint": float(lift_joint),
                "lift_over_marginal": float(lift_over_marginal),
                "log_interaction": float(log_interaction),
            })

        summary_rows.append({
            "pair": key_pair,
            "feature_a": a, "feature_b": b,
            "mi_y_combo": float(mi),
            "n_combos": int(len(keys)),
            "best_combo_a": (best["a_bin"] if best else None),
            "best_combo_b": (best["b_bin"] if best else None),
            "best_combo_imps": (best["imps"] if best else None),
            "best_combo_log_interaction": (best["log_interaction"] if best else None),
        })

    combos = pd.DataFrame(combo_rows)
    # 过滤低曝光组合（避免噪声），并输出 top 组合
    combos = combos[combos["imps"] >= int(cfg.pair_min_imps)].copy()
    combos = combos.sort_values(["pair", "log_interaction"], ascending=[True, False])

    summary = pd.DataFrame(summary_rows).sort_values("mi_y_combo", ascending=False)

    combos.to_parquet(os.path.join(cfg.out_root, "pair_interaction_combos.parquet"), index=False)
    summary.to_parquet(os.path.join(cfg.out_root, "pair_interaction_summary.parquet"), index=False)


# =========================================================
# 总入口：一次跑完 4 个 EDA
# =========================================================
def run_all_eda_extra(
    cfg: AvazuEdaExtraConfig,
    feature_cols: List[str],
) -> None:
    os.makedirs(cfg.out_root, exist_ok=True)

    # 先构建 train top bins（PSI / pair / CTR 候选都会用到）
    top_bins = build_train_top_bins(cfg, feature_cols)

    run_psi_by_field_day(cfg, feature_cols, top_bins=top_bins)
    run_new_value_rate(cfg, feature_cols, top_bins=top_bins)
    run_ctr_by_value_significance(cfg, feature_cols, top_bins=top_bins)
    run_pairwise_interaction(cfg, feature_cols, top_bins=top_bins)
