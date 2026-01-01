# src/cli/save_avazu_adapter_config.py
from __future__ import annotations

import os
from datetime import datetime

from src.data.splits import SplitSpec
from src.data.adapters.avazu import AvazuAdapter
from src.data.adapters.factory import dump_config_from_adapter


def main():
    train_path = r"E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\raw\avazu\train.parquet"
    test_path  = r"E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\raw\avazu\test.parquet"

    split = SplitSpec(
        method="time",
        train_end_day=20141028,
        valid_days=[20141029, 20141030],
        test_day=None,
        test_days=None,
        assert_disjoint=True,
    )

    adp = AvazuAdapter(train_path=train_path, test_path=test_path, split=split)

    # 你想保存到 interim 的 meta
    out_dir = r"E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\interim\avazu\meta"
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"adapter_split_{ts}.yaml")

    cfg = dump_config_from_adapter(
        adp,
        out_path,
        extra={
            "repro": {
                "saved_at": ts,
                "notes": "train=21-28, valid=29-30, external test=31",
            }
        },
    )

    print("saved config to:", out_path)
    print("class_path:", cfg["adapter"]["class_path"])


if __name__ == "__main__":
    main()
