import sys
from types import SimpleNamespace
sys.path.append('ctr_llm_creative')
from src.eda import avazu_evidence_eda as m

# 说明: 干运行，使用小样本设置，安全验证所有阶段（1-9）通路
args = SimpleNamespace(
    input_csv=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\raw\avazu\train.csv',
    out_root='data/interim/avazu',
    chunksize=500000,            # 流式块大小
    topk=20,
    top_values=30,
    min_support=100,
    sample_rows=10000,         # 只采样 10k 行用于部分分析
    seed=42,
    train_days=7,
    test_days=2,
    verbose=False
)

print('Starting dry-run EDA (small sample)...')
try:
    m.main(args)
    print('Dry-run main() completed successfully')
except Exception as e:
    print('Dry-run main() failed:', e)
    raise
