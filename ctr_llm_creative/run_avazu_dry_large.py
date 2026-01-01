import sys
from types import SimpleNamespace
sys.path.append('ctr_llm_creative')
from src.eda import avazu_evidence_eda as m

# Dry-run with larger chunksize for speed (keeps sample_rows small)
args = SimpleNamespace(
    input_csv=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\raw\avazu\train.csv',
    out_root=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\interim\avazu',
    chunksize=2000000,            # larger chunk for speed
    topk=20,
    top_values=30,
    min_support=100,
    sample_rows=10000,         # sample 10k rows for heavy analyses
    seed=42,
    train_days=7,
    test_days=2,
    verbose=False
)

print('Starting dry-run (large chunksize)...')
try:
    m.main(args)
    print('Dry-run completed successfully')
except Exception as e:
    print('Dry-run failed:', e)
    raise
