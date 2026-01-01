# Avazu EDA 运行指南

## 快速对比

| 指标 | 干运行 (Dry-Run) | 全量运行 (Full-Run) |
|------|------------------|------------------|
| **文件** | `run_dry.py` | `run_full.py` |
| **样本大小** | 1K 行 | 整个数据集 (~40M+ 行) |
| **处理模式** | 快速验证管道 | 完整生产分析 |
| **执行时间** | < 1 分钟 | 30-60 分钟 |
| **输出目录** | `data/interim/avazu_dry/` | `data/interim/avazu/` |
| **用途** | 调试、验证管道 | 生产决策、特征设计 |
| **内存占用** | 低（仅 1K 样本） | ~500MB-1GB（流式） |

---

## 1. 干运行（快速验证）

### 何时使用
- ✓ 刚开始：验证整个管道没有错误
- ✓ 调试阶段：快速测试代码更改
- ✓ CI/CD 环节：快速冒烟测试
- ✗ 不用于：生产决策（样本太小，结果不准确）

### 运行命令
```bash
cd ctr_llm_creative
python src/eda/avazu/run_dry.py
```

### 参数
```
chunksize:    500K（标准块大小）
sample_rows:  1K（仅用于 MI/OOV/PSI，其他阶段扫全数据前 1K 行）
topk:         10（简化版，只提取 top 10）
top_values:   10（简化版）
train_days:   1（短时间窗口）
test_days:    1
```

### 输出
```
data/interim/avazu_dry/
├── eda/
│   ├── overview.json
│   ├── schema.json
│   ├── columns_profile.csv
│   ├── time/time_ctr_day.csv
│   ├── hash/hash_collision_sim.csv
│   ├── interactions/pair_mi_topbins.csv
│   └── ...
├── featuremap/featuremap_spec.yml
├── model_plan/model_plan.yml
├── reports/
│   ├── model_structure_evidence.md
│   └── featuremap_evidence.md
└── logs/eda_*.log
```

### 预期结果
- ✓ 所有 9 个阶段都成功运行
- ✓ 没有数据类型或导入错误
- ✓ 能看到 markdown 报告生成
- ⚠ 数据样本太小，结果可能不代表全体特征分布

---

## 2. 全量运行（完整分析）

### 何时使用
- ✓ 生产分析：基于整个数据集做决策
- ✓ 特征工程：设计 FeatureMap 规范
- ✓ 模型选择：基于 MI 互信息选择 DeepFM 还是简单网络
- ✓ 最终上线：所有证据都来自完整数据

### 运行命令
```bash
cd ctr_llm_creative
python src/eda/avazu/run_full.py
```

### 参数
```
chunksize:    1M（平衡速度和内存）
sample_rows:  200K（MI/OOV/PSI 采样，其他全扫）
topk:         20（详细的 top 值）
top_values:   200（详细的 top 值）
train_days:   7（完整时间窗口）
test_days:    2
```

### 输出
```
data/interim/avazu/
├── eda/
│   ├── overview.json                       # 全局统计
│   ├── schema.json                         # 列元数据
│   ├── columns_profile.csv                 # 列特征统计
│   ├── topk/topk_{col}.csv                 # 每列 top K 值
│   ├── time/
│   │   ├── time_ctr_day.csv                # 日粒度 CTR（用于漂移分析）
│   │   └── time_ctr_hour.csv               # 小时粒度 CTR（可选）
│   ├── label/ctr_by_{col}_top.csv          # 特征值分组 CTR
│   ├── split/oov_rate_train_test.csv       # OOV 检测
│   ├── drift/
│   │   ├── psi_train_test_summary.csv      # PSI 汇总
│   │   └── psi_train_test_{col}.csv        # 每列 PSI 详情
│   ├── hash/hash_collision_sim.csv         # Hash 碰撞率 → 选 bucket 大小
│   ├── interactions/pair_mi_topbins.csv    # 特征交互强度 → 选模型（DeepFM vs DNN）
│   └── leakage/leakage_signals.csv         # 泄漏风险信号
├── featuremap/featuremap_spec.yml          # FeatureMap 规范（直接用于模型）
├── model_plan/model_plan.yml               # 模型架构推荐
├── reports/
│   ├── model_structure_evidence.md         # 基于 MI 的模型选择证据链
│   └── featuremap_evidence.md              # 特征工程决策链
└── logs/eda_*.log                          # 详细运行日志
```

### 输出文件关键指标

#### 1. `overview.json` - 全局统计
```json
{
  "total_rows": 40521573,
  "total_clicks": 6865257,
  "global_ctr": 0.1694,
  "columns": 24,
  "null_counts": { ... }
}
```
**用途**：基准 CTR、数据规模

#### 2. `columns_profile.csv` - 列特征统计
```
column,dtype,null_rate,nunique,top1_value,top1_rate,entropy,hhi,is_id_like
id,str,0.00,40521573,1,0.0,NA,NA,True        # 高基数，ID 类
click,int,0.00,2,1,0.169,0.568,0.331,False   # 标签列
hour,str,0.00,1463,20200625,0.0024,10.24,0.0002,False  # 时间列
site_id,str,0.00,4737,85f69d40,0.011,9.16,0.0015,False # 低基数，分类
...
```
**用途**：
- `null_rate` > 0.1 → 需要处理缺失
- `nunique` > 100K → 高基数，需要 hash 或嵌入
- `hhi` > 0.01 → 长尾分布，可能有 OOV 风险
- `is_id_like` = True → 应该单独处理或丢弃

#### 3. `time_ctr_day.csv` - 日粒度 CTR（时间漂移分析）
```
day,impressions,clicks,ctr
20200625,1000000,169000,0.169
20200626,1050000,177000,0.1686
...
```
**用途**：
- CTR 变化趋势 → 选择 train/test 分割点
- PSI 计算输入
- 时间漂移检测

#### 4. `ctr_by_{col}_top.csv` - 特征值分组 CTR
```
value,impressions,clicks,ctr,ctr_ci_lower,ctr_ci_upper,z_score,lift
2,150000,30000,0.200,0.198,0.202,2.5,1.18      # lift = 0.200 / 0.169 = 1.18
3,200000,32000,0.160,0.158,0.162,-1.8,0.95
...
```
**用途**：
- 特征值是否有明显的 CTR 差异 → 信息量
- `lift` > 1.1 或 < 0.9 → 强特征
- `z_score` 显著性检验

#### 5. `hash_collision_sim.csv` - Hash 碰撞率
```
buckets,collision_rate,expected_uniques,estimation_method
262144,0.0045,256234,md5_count
524288,0.0022,256234,md5_count
1048576,0.0011,256234,md5_count
2097152,0.0005,256234,md5_count
```
**用途**：
- 选择最小的 bucket 大小使得碰撞率 < 0.5%
- 例如：选 524K 碰撞率 0.22% 就可以（不用 1M）
- 直接影响 embedding 层大小

#### 6. `pair_mi_topbins.csv` - 特征交互强度（MI）
```
col_a,col_b,mi,support_pairs,notes
site_id,app_id,1.576,50000,top_bins=30|30
site_category,app_category,0.999,80000,top_bins=20|15
hour,site_id,0.234,100000,top_bins=10|30
device_id,user_agent,0.156,10000,top_bins=5|5
...
```
**用途**：
- **平均 MI > 0.005** 或 **强交互对 > 3 个** → 推荐 **DeepFM 或 DCN**
- **平均 MI < 0.001** → 推荐 **简单 DNN**（互信息弱，交互学习价值小）
- 这是 **模型选型的关键决策点**

#### 7. `psi_train_test_summary.csv` - 分布漂移 PSI
```
column,psi,risk_level,description
hour,0.35,HIGH,Strong time drift detected
site_id,0.08,LOW,Stable distribution
app_id,0.12,MEDIUM,Moderate drift
...
```
**用途**：
- PSI < 0.1 → 低风险，正常使用
- PSI 0.1-0.25 → 中等风险，注意监控
- PSI > 0.25 → 高风险，考虑样本重加权或重训练

#### 8. `oov_rate_train_test.csv` - OOV 率（Out-of-Vocabulary）
```
column,oov_rate_pct,new_values_in_test,coverage_rate
site_id,2.1%,120,97.9%
app_id,1.8%,90,98.2%
device_id,5.3%,1000,94.7%
...
```
**用途**：
- OOV > 5% → 需要 embedding 层的 OOV 处理（稀疏向量或特殊 token）
- OOV < 1% → 可以直接用 embedding 层（很少遇到 OOV）

#### 9. `model_structure_evidence.md` - 模型选择证据链
```markdown
# Model Structure Evidence & Recommendations

## Decision: DeepFM vs Simple DNN

### Metric: Pairwise Mutual Information (MI)
- Average MI: 0.087
- Strong pairs (MI > 0.01): 42 pairs
- Weak pairs (MI < 0.001): 156 pairs

### Evidence
From interactions/pair_mi_topbins.csv:
- site_id × app_id: MI = 1.576 (VERY STRONG)
- site_category × app_category: MI = 0.999 (STRONG)
- ...

### Recommendation: ✓ DeepFM
Reasoning: Average MI = 0.087 >> 0.005 threshold
The model should focus on learning feature interactions.
```

#### 10. `featuremap_spec.yml` - FeatureMap 规范（直接用于模型）
```yaml
features:
  categorical:
    site_id:
      type: "categorical"
      vocab_size: 5000
      embedding_dim: 8
      hash_bucket_size: 524288  # 从 hash_collision_sim.csv 选择
      oov_handling: "sparse_unknown"
    
    app_id:
      type: "categorical"
      vocab_size: 4000
      embedding_dim: 8
      hash_bucket_size: 524288
      oov_handling: "sparse_unknown"
    
    hour:
      type: "temporal_categorical"
      vocab_size: 1463  # 独特天数
      embedding_dim: 4
      normalization: None
  
  numerical:
    price:
      type: "numerical"
      normalization: "log"  # 基于列特征统计决定

model_architecture:
  type: "deepfm"  # 基于 MI 互信息决定
  use_fm_layer: true
  use_deep_layer: true
  deep_hidden_dims: [256, 128, 64]
```

---

## 3. 对比流程

### 干运行 → 全量运行 工作流

```
1. 开始阶段：运行干运行验证管道
   ↓
2. 检查 data/interim/avazu_dry/ 目录
   ├─ 是否有所有输出文件？
   ├─ model_structure_evidence.md 是否能打开？
   └─ logs 中是否有错误？
   ↓
3. 如果都 OK，运行全量运行
   ↓
4. 等待 30-60 分钟...
   ↓
5. 检查 data/interim/avazu/ 目录
   ├─ 检查 pair_mi_topbins.csv：平均 MI 多少？
   ├─ 检查 hash_collision_sim.csv：选哪个 bucket 大小？
   └─ 检查 featuremap_spec.yml：是否可用？
   ↓
6. 根据生成的 featuremap_spec.yml 和 model_plan.yml 训练模型
```

---

## 4. 关键决策点

### Q1: 选择 DeepFM 还是简单 DNN？
**答**: 查看 `pair_mi_topbins.csv`
- 平均 MI > 0.005 或强对 > 3 个 → **DeepFM**
- 平均 MI < 0.001 → **简单 DNN**（不学交互）

### Q2: Hash Bucket 大小选多少？
**答**: 查看 `hash_collision_sim.csv`
- 选碰撞率 < 0.5% 的最小值
- 一般是 262K 或 512K

### Q3: 特征是否有分布漂移？
**答**: 查看 `psi_train_test_summary.csv`
- PSI > 0.25 → 高风险，需要处理
- PSI < 0.1 → 低风险

### Q4: OOV 需要特殊处理吗？
**答**: 查看 `oov_rate_train_test.csv`
- OOV > 5% → 需要 embedding layer 的特殊处理
- OOV < 1% → 直接用

---

## 5. 常见问题

### Q: 为什么干运行数据这么少（1K）？
**A**: 干运行的目的是快速验证管道没有 bug，不是得到准确结果。生产决策必须用全量运行。

### Q: 干运行和全量运行能并行吗？
**A**: 可以，因为输出目录不同（`avazu_dry` vs `avazu`）。

### Q: 全量运行太慢了怎么办？
**A**: 
- 增大 `chunksize`（默认 1M，可改 2M）
- 减少 `topk` 或 `top_values`
- 在 SSD 快速磁盘上运行

### Q: 能只运行某些阶段吗？
**A**: 需要修改 `src/eda/avazu_evidence_eda.py` 的 `main()` 函数，注释掉不需要的阶段。

---

## 6. 输出文件大小估算

| 文件 | 干运行 | 全量运行 |
|------|--------|---------|
| `overview.json` | < 1 KB | < 1 KB |
| `columns_profile.csv` | 50 KB | 50 KB |
| `topk/*.csv` | 5 MB | 100 MB |
| `time_ctr_day.csv` | 10 KB | 500 KB |
| `ctr_by_*.csv` | 20 MB | 1 GB |
| `hash_collision_sim.csv` | 2 KB | 2 KB |
| `pair_mi_topbins.csv` | 1 MB | 50 MB |
| `psi_train_test_*.csv` | 5 MB | 200 MB |
| **总计** | ~30 MB | ~1.5 GB |

---

## 7. 下一步

1. **运行干运行**
   ```bash
   python src/eda/avazu/run_dry.py
   ```

2. **检查输出**
   ```
   ls data/interim/avazu_dry/eda/
   cat data/interim/avazu_dry/reports/model_structure_evidence.md
   ```

3. **如果成功，运行全量运行**
   ```bash
   python src/eda/avazu/run_full.py  # 等 30-60 分钟
   ```

4. **基于输出设计 FeatureMap 和模型**
   - 参考 `data/interim/avazu/featuremap/featuremap_spec.yml`
   - 参考 `data/interim/avazu/reports/model_structure_evidence.md`
