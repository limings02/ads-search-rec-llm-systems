# Avazu EDA Scripts

Quick-start scripts for running Avazu CTR dataset evidence-driven EDA.

## 文件

- **run_dry.py** - 干运行（快速验证，10K 采样）
- **run_full.py** - 全量运行（完整分析）

## 干运行（快速验证）

快速验证所有 9 个 EDA 阶段通路，10K 采样，2-5 分钟内完成：

```bash
python src/eda/avazu/run_dry.py
```

**参数**：
- chunksize: 200K（200K 行的块读取）
- sample_rows: 10K（采样 10K 行用于 MI/OOV/PSI）
- topk: 20（每个特征提取 20 个 top 值）
- top_values: 30（CTR 分析使用 30 个 top 值）

**输出**：所有关键产物保存到 `data/interim/avazu/`（目录结构如下）

## 全量运行（完整分析）

在完整数据集上运行所有 9 个 EDA 阶段，生成完整决策链：

```bash
python src/eda/avazu/run_full.py
```

**参数**：
- chunksize: 500K（500K 行的块读取，生产级）
- sample_rows: 200K（采样 200K 行用于 MI/OOV/PSI）
- topk: 20
- top_values: 200（详细的 CTR 分析）

**执行时间**：30-60 分钟（取决于数据集大小，train.csv 通常 40M+ 行）

**内存**：~500MB-1GB（流式处理，安全）

## 输出目录结构

```
data/interim/avazu/
├── eda/
│   ├── overview.json                    # 全局统计
│   ├── schema.json                      # 列元数据
│   ├── columns_profile.csv              # 列描述统计
│   ├── topk/
│   │   └── topk_{col}.csv               # 每列 top K 值
│   ├── time/
│   │   ├── time_ctr_hour.csv            # 日粒度 CTR 趋势
│   │   └── time_ctr_day.csv
│   ├── label/
│   │   └── ctr_by_{col}_top.csv         # 特征分组 CTR + CI
│   ├── split/
│   │   └── oov_rate_train_test.csv      # OOV 检测
│   ├── drift/
│   │   └── psi_train_test_summary.csv   # PSI 漂移分析
│   ├── hash/
│   │   └── hash_collision_sim.csv       # Hash 碰撞率（Bucket 大小选择）
│   ├── interactions/
│   │   └── pair_mi_topbins.csv          # 特征交互强度 (MI)
│   └── leakage/
│       └── leakage_signals.csv
├── featuremap/
│   └── featuremap_spec.yml              # FeatureMap 设计建议
├── model_plan/
│   └── model_plan.yml
└── reports/
    ├── model_structure_evidence.md      # 模型架构推荐（基于 MI）
    └── featuremap_evidence.md           # 特征工程决策链

logs/                                    # EDA 日志
├── eda_*.log                            # 运行日志
```

## 关键文件说明

### 1. `overview.json` - 全局统计
总行数、全局 CTR、缺失率、异常值等

### 2. `columns_profile.csv` - 列特征统计
每列的缺失率、基数、top1 占比、熵、HHI、ID 相似度

### 3. `time_ctr_day.csv` - 日粒度 CTR
按日期聚合的 CTR（用于时间切分与漂移检测）

### 4. `ctr_by_{col}_top.csv` - 特征分组 CTR
- 按特征值分组的 CTR
- Wilson 置信区间（95%）
- Lift（相对于全局 CTR）
- Z-score 显著性检验

### 5. `hash_collision_sim.csv` - Hash 碰撞率
5 种 bucket 大小 (2^18 ~ 2^22) 的碰撞率估计
→ **决策**：选择碰撞率 < 0.5% 的最小 bucket 大小

### 6. `pair_mi_topbins.csv` - 特征交互 MI
top 特征对的互信息（交互强度代理）
→ **决策**：MI > 0.01 的对数多，建议用 DeepFM/DCN；否则简单网络即可

### 7. `psi_train_test_summary.csv` - 分布漂移 PSI
PSI > 0.25 = 高漂移，需要关注或重训练

### 8. `oov_rate_train_test.csv` - OOV 率
列的 OOV 率（新值占比）→ 指导嵌入层处理策略

### 9. `model_structure_evidence.md` - 模型选型报告
基于 MI 互信息推荐模型结构（DeepFM vs 简单 DNN）

### 10. `featuremap_spec.yml` - FeatureMap 规范
- Hash bucket 大小（基于 hash_collision_sim.csv）
- 词汇表大小（基于 columns_profile.csv）
- 嵌入维度建议

## 工作流

```
run_dry.py (快速验证)
    ↓
验证所有 9 个阶段通路无误
    ↓
run_full.py (完整分析)
    ↓
读取 data/interim/avazu/ 中的各个 CSV/JSON
    ↓
特征工程设计：
  - Hash bucket 大小 ← hash_collision_sim.csv
  - 词汇表大小 ← columns_profile.csv (nunique)
  - OOV 处理 ← oov_rate_train_test.csv
  - 特征交互 ← pair_mi_topbins.csv
    ↓
模型选型：
  - DeepFM (if avg MI > 0.005) ← pair_mi_topbins.csv
  - 简单 DNN (if avg MI < 0.001)
  - 根据 PSI 判断是否需要样本重加权 ← psi_train_test_summary.csv
    ↓
生成 FeatureMap YAML（src/eda/avazu/featuremap_spec.yml）
    ↓
传入模型训练管道
```

## 常见问题

### Q1: 干运行失败，提示 "hour must be 8-digit"
**A**: 数据的 hour 字段是 YYYYMMDD（8 位），不是 YYYYMMDDHH（10 位）。代码已修正。

### Q2: 整个流程耗时太长
**A**: 
- 减少 `sample_rows`（默认 200K，改为 50K）
- 增大 `chunksize`（默认 500K，改为 1M，但注意内存）
- 跳过不需要的阶段（修改 main() 中的阶段注释）

### Q3: 输出文件太大
**A**: 
- `columns_profile.csv` 和 topk CSV 可能 100MB+（有 topk 数据）
- 可删除不需要的 topk_{col}.csv 文件（保留 columns_profile.csv 和 topk_day 等核心文件）
- 或增加 `topk` 参数从 20 改为 10

### Q4: 如何自定义参数？
**A**: 编辑 `run_dry.py` 或 `run_full.py` 中的 `args` 字段：
```python
args = SimpleNamespace(
    chunksize=1000000,          # 增大到 1M
    sample_rows=500000,         # 增加采样
    topk=50,                    # 更多 top 值
    verbose=True,               # 详细日志
    ...
)
```

## 下一步

1. **干运行验证**（5 分钟）：`python src/eda/avazu/run_dry.py`
2. **检查输出**：查看 `data/interim/avazu/reports/model_structure_evidence.md`
3. **全量运行**（30-60 分钟）：`python src/eda/avazu/run_full.py`
4. **生成 FeatureMap**：加载 `data/interim/avazu/featuremap/featuremap_spec.yml`
5. **模型训练**：使用生成的 FeatureMap 和推荐架构进行训练

---

**作者**: Evidence-driven EDA Pipeline  
**最后更新**: 2024  
**版本**: Stage 1-9 完整实现
