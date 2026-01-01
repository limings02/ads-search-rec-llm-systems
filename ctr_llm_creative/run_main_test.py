from types import SimpleNamespace
import sys
sys.path.append('ctr_llm_creative')
from src.eda import avazu_evidence_eda as m

args = SimpleNamespace(
    input_csv='test_output/synth.csv',
    out_root='test_output',
    chunksize=5000,
    topk=20,
    top_values=30,
    min_support=100,
    sample_rows=20000,
    seed=42,
    train_days=7,
    test_days=2,
    verbose=False
)

print('Running main with synthetic data...')
m.main(args)
print('MAIN_DONE')
