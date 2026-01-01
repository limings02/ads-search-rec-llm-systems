from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Iterable, Tuple

import pandas as pd

from .base_adapter import BaseAdapter
from src.data.splits import SplitSpec


class AvazuAdapter(BaseAdapter):
    """
    Avazu Parquet Adapter (train/valid from train.parquet, external test from test.parquet).

    统一输出一套“标准 Avazu 特征列”，保证 train/valid/test 三者列完全一致：
      - C1
      - banner_pos, site_id, site_domain, site_category,
        app_id, app_domain, app_category,
        device_id, device_ip, device_model, device_type, device_conn_type
      - C14..C21

    兼容两种输入风格：
      A) 标准具名列（你的 test.parquet 就是这种）
      B) 预处理成 C2..C13 的风格（若你的 train.parquet 是这种，会自动映射回具名列）
    """

    name = "avazu"

    # Avazu 标准特征列（强烈建议你 FeatureMap/EDA/训练都基于这套名字）
    AVAZU_NAMED_COLS = [
        "banner_pos",
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category",
        "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    ]
    AVAZU_C_COLS = ["C1"] + [f"C{i}" for i in range(14, 22)]
    AVAZU_FEATURE_COLS = AVAZU_C_COLS + AVAZU_NAMED_COLS  # 共 1 + 8 + 12 = 21 列

    # 若输入是 C2..C13 风格，将其映射回标准具名列
    C2C13_TO_NAMED = {
        "C2": "banner_pos",
        "C3": "site_id",
        "C4": "site_domain",
        "C5": "site_category",
        "C6": "app_id",
        "C7": "app_domain",
        "C8": "app_category",
        "C9": "device_id",
        "C10": "device_ip",
        "C11": "device_model",
        "C12": "device_type",
        "C13": "device_conn_type",
    }

    def __init__(
        self,
        raw_path: Optional[str] = None,        # 兼容旧调用
        split: Optional[SplitSpec] = None,
        train_path: Optional[str] = None,      # 推荐参数名
        test_path: Optional[str] = None,
        usecols: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
    ):
        if train_path is None:
            train_path = raw_path
        if train_path is None:
            raise ValueError("AvazuAdapter requires train_path (or raw_path).")
        if split is None:
            raise ValueError("AvazuAdapter requires split: SplitSpec(...).")

        # 默认读取：核心列 + 标准特征列 + 兼容列(C2..C13)
        if usecols is None:
            usecols = ["id", "click", "hour"] \
                      + self.AVAZU_C_COLS \
                      + self.AVAZU_NAMED_COLS \
                      + list(self.C2C13_TO_NAMED.keys())  # C2..C13

        super().__init__(raw_path=train_path, split=split, usecols=usecols, dtypes=dtypes or {}, encoding=None)

        self.train_path = self._resolve_parquet(train_path, default_name="train.parquet")
        self.test_path = self._resolve_parquet(test_path, default_name="test.parquet") if test_path else None

        self._global_row_offset_train = 0
        self._global_row_offset_test = 0

    @staticmethod
    def _resolve_parquet(path: str, default_name: str) -> str:
        if os.path.isfile(path):
            return path
        candidate = os.path.join(path, default_name)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Cannot find parquet: {path} (or {candidate})")

    # -------------------------
    # 读取 train.parquet（train/valid）
    # -------------------------
    def iter_raw_chunks(self, chunksize: int) -> Iterator[pd.DataFrame]:
        yield from self._iter_parquet_batches(
            parquet_path=self.train_path,
            chunksize=chunksize,
            requested_cols=self.usecols,
            global_offset_attr="_global_row_offset_train",
            allow_missing_columns=True,   # 关键：允许 train parquet 缺部分特征列（但 click 仍在 canonicalize 强制）
        )

    # -------------------------
    # 读取外置 test.parquet（test）
    # -------------------------
    def iter_test_chunks(self, chunksize: int) -> Iterator[pd.DataFrame]:
        if not self.test_path:
            raise ValueError("test_path is None. Provide test_path to load external test set.")
        yield from self._iter_parquet_batches(
            parquet_path=self.test_path,
            chunksize=chunksize,
            requested_cols=self.usecols,
            global_offset_attr="_global_row_offset_test",
            allow_missing_columns=True,   # 允许 test 缺 click / 缺部分特征
        )

    def _iter_parquet_batches(
        self,
        parquet_path: str,
        chunksize: int,
        requested_cols: Optional[List[str]],
        global_offset_attr: str,
        allow_missing_columns: bool,
    ) -> Iterator[pd.DataFrame]:
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise ImportError("Streaming parquet requires pyarrow. Install: pip install pyarrow") from e

        pf = pq.ParquetFile(parquet_path)
        schema_cols = set(pf.schema_arrow.names)

        cols = None
        if requested_cols is not None:
            cols = [c for c in requested_cols if c in schema_cols]
            if (not allow_missing_columns) and (len(cols) != len(requested_cols)):
                missing = [c for c in requested_cols if c not in schema_cols]
                raise ValueError(f"Parquet missing columns (strict mode): {missing}. file={parquet_path}")

        for batch in pf.iter_batches(batch_size=chunksize, columns=cols):
            df = batch.to_pandas()

            # 如果没有 id，就造全局行号（chunk 间唯一）
            if "id" not in df.columns:
                n = len(df)
                off = getattr(self, global_offset_attr)
                df["__row_id"] = pd.RangeIndex(off, off + n).astype("int64")
                setattr(self, global_offset_attr, off + n)

            yield df

    # -------------------------
    # canonicalize：训练默认严格（必须有 click）
    # -------------------------
    def canonicalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self._canonicalize(raw_df, require_click_label=True)

    def _canonicalize(self, raw_df: pd.DataFrame, require_click_label: bool) -> pd.DataFrame:
        df = raw_df.copy()

        # 1) _id
        if "id" in df.columns:
            df["_id"] = df["id"].astype("string")
        elif "__row_id" in df.columns:
            df["_id"] = df["__row_id"].astype("int64").astype("string")
        else:
            df["_id"] = df.index.astype("int64").astype("string")

        # 2) label
        if "click" in df.columns:
            df["_label_click"] = df["click"].astype("int8")
        else:
            if require_click_label:
                raise ValueError("Missing 'click' label in input, but require_click_label=True.")
            df["_label_click"] = pd.Series([pd.NA] * len(df), index=df.index)

        # 3) hour -> _day/_hod
        if "hour" not in df.columns:
            raise ValueError("Missing required column: hour")
        day, hod = self._parse_hour_to_day_hod(df["hour"])
        df["_day"] = day
        df["_hod"] = hod

        # 4) 兼容：若输入是 C2..C13 风格，映射到具名列
        for c_col, named_col in self.C2C13_TO_NAMED.items():
            if (named_col not in df.columns) and (c_col in df.columns):
                df[named_col] = df[c_col]

        # 5) 统一补齐所有标准特征列（缺的补 __MISSING__）
        for col in self.AVAZU_FEATURE_COLS:
            if col not in df.columns:
                df[col] = "__MISSING__"
            else:
                df[col] = df[col].fillna("__MISSING__").astype("string")

        # 6) 输出列裁剪：只保留稳定的 canonical 列（强烈建议）
        out_cols = ["_id", "_day", "_hod", "_label_click"] + self.AVAZU_FEATURE_COLS
        return df[out_cols]

    @staticmethod
    def _parse_hour_to_day_hod(hour_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
        s = hour_col.astype("string").str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        s = s.str.zfill(8)

        sample = s.dropna().head(1)
        if len(sample) == 0:
            raise ValueError("hour column empty after parsing")
        L = len(sample.iloc[0])

        if L >= 10:
            yyyy = s.str.slice(0, 4).astype("int32")
            mm = s.str.slice(4, 6).astype("int32")
            dd = s.str.slice(6, 8).astype("int32")
            hod = s.str.slice(8, 10).astype("int16")
        else:
            yy = s.str.slice(0, 2).astype("int32")
            mm = s.str.slice(2, 4).astype("int32")
            dd = s.str.slice(4, 6).astype("int32")
            hod = s.str.slice(6, 8).astype("int16")
            yyyy = (2000 + yy).astype("int32")

        day = (yyyy * 10_000 + mm * 100 + dd).astype("int32")
        return day, hod

    # -------------------------
    # 覆盖 iter_splits：把外置 test 也 yield 出来
    # -------------------------
    def iter_splits(self, chunksize: int) -> Iterator[Tuple[str, pd.DataFrame]]:
        # train/valid：来自 train.parquet，按 SplitSpec 切分
        for raw in self.iter_raw_chunks(chunksize=chunksize):
            cdf = self.canonicalize(raw)  # 严格：必须有 click
            masks = self.split_masks(cdf)
            for split_name, mask in masks.items():
                part = cdf[mask]
                if len(part) > 0:
                    yield split_name, part

        # external test：来自 test.parquet，不要求 click
        if self.test_path:
            for raw in self.iter_test_chunks(chunksize=chunksize):
                cdf = self._canonicalize(raw, require_click_label=False)
                yield "test", cdf

    def get_features(self) -> List[str]:
        """下游 FeatureEngineering/EDA 只认这套稳定特征列。"""
        return self.AVAZU_FEATURE_COLS
