# src/data/adapters/base_adapter.py
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Literal
import pandas as pd

# SplitSpec 与切分实现放在 src/data/splits.py（可复用、可单测）
from src.data.splits import SplitSpec, make_split_masks


DatasetName = Literal["avazu", "ali_ccp", "ipinyou", "criteo_attr"]


class BaseAdapter:
    """
    BaseAdapter = “编排层”（orchestration），不是“算法层”。

    它负责把一条数据管道统一成：
      raw -> canonical -> split masks -> (train/valid/test chunk)

    子类（AvazuAdapter / AliCCPAdapter / IPinYouAdapter / CriteoAttrAdapter）只负责两件事：
    1) iter_raw_chunks：怎么读文件/多文件/tsv/csv/parquet
    2) canonicalize   ：把原始列映射到统一列（至少 _id/_day/labels）

    注意：BaseAdapter 不做 hashing/分桶/embedding —— 这些属于 feature_engineering。
    """

    name: DatasetName = "avazu"

    def __init__(
        self,
        raw_path: str,
        split: SplitSpec,
        usecols: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
        encoding: Optional[str] = None,
    ):
        self.raw_path = raw_path
        self.split = split
        self.usecols = usecols
        self.dtypes = dtypes
        self.encoding = encoding

    # -------- 子类必须实现：读取 raw --------
    def iter_raw_chunks(self, chunksize: int) -> Iterator[pd.DataFrame]:
        """
        产出 raw DataFrame chunk。
        - Avazu：单个 train.csv
        - Ali-CCP：可能是多个 tsv 文件（你可以在这里 glob 多文件逐个读）
        - iPinYou：通常也是多文件日志
        - Criteo Attr：可能是 parquet 或多文件
        """
        raise NotImplementedError

    # -------- 子类必须实现：标准化字段 --------
    def canonicalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        把 raw_df 转成 canonical_df，并保证：
        - _id: 稳定唯一 id（random split / debug）
        - _day: YYYYMMDD（time/day split；强烈建议所有数据集都产出）
        - label 列统一命名（建议 _label_click/_label_conv/...）
        """
        raise NotImplementedError

    # -------- 通用：切分（调用 src/data/splits.py）--------
    def split_masks(self, canonical_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        这里不写切分算法细节，只做调用与返回统一结构。
        切分算法在 src/data/splits.py，便于复用和单测。
        """
        return make_split_masks(canonical_df, self.split)

    # -------- 通用：给下游统一消费（流式）--------
    def iter_splits(self, chunksize: int) -> Iterator[Tuple[str, pd.DataFrame]]:
        """
        给 Trainer/EDA 的统一入口：
        for split_name, df in adapter.iter_splits(...):
            if split_name == 'train': partial_fit / train_step
            else: accumulate metrics
        """
        for raw in self.iter_raw_chunks(chunksize=chunksize):
            cdf = self.canonicalize(raw)
            masks = self.split_masks(cdf)
            for split_name, mask in masks.items():
                part = cdf[mask]
                if len(part) > 0:
                    yield split_name, part
