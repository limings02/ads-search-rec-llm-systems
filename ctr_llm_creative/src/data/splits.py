# src/data/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Literal

import numpy as np
import pandas as pd
import hashlib


SplitMethod = Literal["none", "time", "day", "random"]


@dataclass
class SplitSpec:
    """
    支持四个数据集统一切分的配置。

    重点改动：支持“混合配置”
    - train 用 train_end_day（<=）
    - valid 用 valid_days（isin 多天）或 valid_day（单天）
    - test  用 test_days / test_day（可选）

    这样你就能表达：
      train: 20141021~20141028
      valid: 20141029 + 20141030
      test : （空，或由单独 test_path 提供）
    """
    method: SplitMethod = "time"

    # canonical 列名
    day_col: str = "_day"
    id_col: str = "_id"

    # ---- A) contiguous train（最常用）----
    train_end_day: Optional[int] = None

    # ---- B) explicit sets（更灵活）----
    train_days: Optional[Sequence[int]] = None

    # valid：既支持单天，也支持多天
    valid_day: Optional[int] = None
    valid_days: Optional[Sequence[int]] = None

    # test：既支持单天，也支持多天
    test_day: Optional[int] = None
    test_days: Optional[Sequence[int]] = None

    # ---- C) stable random split（sanity check 用）----
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    hash_key: str = "split_v1"

    assert_disjoint: bool = True


def _assert_disjoint(train_m: pd.Series, valid_m: pd.Series, test_m: pd.Series, where: str) -> None:
    tv = int((train_m & valid_m).sum())
    tt = int((train_m & test_m).sum())
    vt = int((valid_m & test_m).sum())
    if tv or tt or vt:
        raise ValueError(
            f"[split disjoint failed @ {where}] overlap: "
            f"train&valid={tv}, train&test={tt}, valid&test={vt}"
        )


def make_split_masks(df: pd.DataFrame, spec: SplitSpec) -> Dict[str, pd.Series]:
    """
    返回 train/valid/test 的 mask（与 df index 对齐）。
    """
    n = len(df)
    idx = df.index

    if spec.method == "none":
        return {
            "train": pd.Series(np.ones(n, dtype=bool), index=idx),
            "valid": pd.Series(np.zeros(n, dtype=bool), index=idx),
            "test": pd.Series(np.zeros(n, dtype=bool), index=idx),
        }

    if spec.method in ("time", "day"):
        if spec.day_col not in df.columns:
            raise ValueError(
                f"time/day split requires day_col='{spec.day_col}' in canonical_df. "
                f"Please create it in adapter.canonicalize()."
            )
        day = df[spec.day_col].astype("int64")

        # -------- train mask（优先 explicit train_days，其次 train_end_day）--------
        if spec.train_days is not None:
            train_m = day.isin(set(spec.train_days))
        elif spec.train_end_day is not None:
            train_m = day <= int(spec.train_end_day)
        else:
            raise ValueError("time/day split requires either train_days or train_end_day.")

        # -------- valid mask（优先 valid_days，其次 valid_day，否则空）--------
        if spec.valid_days is not None:
            valid_m = day.isin(set(spec.valid_days))
        elif spec.valid_day is not None:
            valid_m = day == int(spec.valid_day)
        else:
            valid_m = day == -1  # empty

        # -------- test mask（优先 test_days，其次 test_day，否则空）--------
        if spec.test_days is not None:
            test_m = day.isin(set(spec.test_days))
        elif spec.test_day is not None:
            test_m = day == int(spec.test_day)
        else:
            test_m = day == -1  # empty

        if spec.assert_disjoint:
            _assert_disjoint(train_m, valid_m, test_m, where="make_split_masks(time/day)")

        return {"train": train_m, "valid": valid_m, "test": test_m}

    if spec.method == "random":
        if spec.id_col not in df.columns:
            raise ValueError(
                f"random split requires id_col='{spec.id_col}' in canonical_df. "
                f"Please create it in adapter.canonicalize()."
            )

        tr = float(spec.train_ratio)
        vr = float(spec.valid_ratio)
        te = float(spec.test_ratio)
        s = tr + vr + te
        if not np.isclose(s, 1.0):
            raise ValueError(f"random split ratios must sum to 1.0, got {s}")

        ids = df[spec.id_col].astype("string").fillna("__NULL__")

        # 稳定 hash -> [0,1)
        def _u01(x: str) -> float:
            h = hashlib.md5((spec.hash_key + "|" + x).encode("utf-8")).hexdigest()
            v = int(h[:8], 16)
            return v / 2**32

        u = ids.map(_u01).astype("float64")
        train_m = u < tr
        valid_m = (u >= tr) & (u < tr + vr)
        test_m = u >= (tr + vr)

        if spec.assert_disjoint:
            _assert_disjoint(train_m, valid_m, test_m, where="make_split_masks(random)")

        return {
            "train": pd.Series(train_m, index=idx),
            "valid": pd.Series(valid_m, index=idx),
            "test": pd.Series(test_m, index=idx),
        }

    raise ValueError(f"unsupported split method: {spec.method}")
