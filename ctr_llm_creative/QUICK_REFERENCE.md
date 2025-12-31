# 快速参考卡片 - CTR/LLM 搜索推荐系统

## 📋 项目架构总览

### 三层架构
```
数据层 (contracts/) 
    ↓
平台层 (src/core/) 
    ↓
应用层 (src/cli/, src/models/, src/sim/, src/analysis/)
```

### 三阶段流程
```
Stage 1: 训练        Stage 2: 仿真        Stage 3: 评估
(predict)        (simulate)        (evaluate)
    ↓                 ↓                  ↓
离线指标            仿真KPI            显著性检验
(AUC, Loss)       (spend, CTR)     (Bootstrap CI)
```

## 🎯 核心契约类（contracts/）

| 类 | 功能 | 用途 |
|----|------|------|
| `DatasetManifest` | 数据集定义 | 描述数据的结构、任务、分割 |
| `FeatureMap` | 特征映射 | 定义特征变换（hash/vocab/bucket） |
| `AuctionStream` | 拍卖流 | 可回放的竞价事件序列 |
| `Metrics` | 指标容器 | 离线、仿真、统计检验结果 |
| `RunMeta` | 运行元数据 | Git信息、Config、环境、seed |

## 📁 关键目录映射

| 目录 | 职责 |
|------|------|
| `contracts/` | 数据结构定义（不含逻辑） |
| `src/core/` | 通用基础设施 |
| `src/cli/` | 命令行入口 |
| `src/data/` | 数据加载与处理 |
| `src/models/` | 模型实现 |
| `src/trainers/` | 训练循环 |
| `src/sim/` | 竞价仿真 |
| `src/analysis/` | 统计分析 |
| `src/api/` | 后端API |
| `configs/` | Hydra配置 |
| `runs/` | 运行输出（自动生成） |

## 🔧 常用命令

### 训练
```bash
python -m src.cli.train --config configs/experiments/avazu_infra_deepfm.yaml
```

### 仿真
```bash
python -m src.cli.simulate --run-id RUN_ID --budget 10000.0
```

### 评估
```bash
python -m src.cli.evaluate --run-id RUN_ID --baseline-run-id BASELINE_ID --alpha 0.05
```

### 导出
```bash
python -m src.cli.export_run --run-id RUN_ID --output result.tar.gz
```

### 测试
```bash
pytest tests/ -v --cov=src/
```

### API
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

## 📊 配置文件层次

```
configs/
├── _base/base.yaml              # 全局基础配置
├── datasets/*.yaml              # 数据集特定配置
├── features/*.yaml              # 特征工程配置
├── models/*.yaml                # 模型超参配置
├── simulation/*.yaml            # 仿真配置
├── evaluation/*.yaml            # 评估配置
└── experiments/*.yaml           # 实验组合（Hydra compose）
```

### 配置例子
```bash
# 使用基础实验配置
python -m src.cli.train \
    --config configs/experiments/avazu_infra_deepfm.yaml

# 命令行override
python -m src.cli.train \
    --config configs/experiments/avazu_infra_deepfm.yaml \
    training.epochs=50 \
    training.batch_size=256 \
    seed=123
```

## 🏗️ 添加新组件的流程

### 添加新数据集适配器
```python
# 1. 创建 src/data/adapters/my_dataset.py
from src.data.adapters.base import BaseAdapter

class MyDatasetAdapter(BaseAdapter):
    def load_split(self, split: str):
        pass
    def get_features(self):
        pass

# 2. 注册到 src/core/registry.py
from src.core.registry import dataset_adapters
dataset_adapters.register("my_dataset", MyDatasetAdapter)

# 3. 创建配置 configs/datasets/my_dataset.yaml
# 4. 编写测试 tests/test_my_dataset_adapter.py
```

### 添加新模型
```python
# 1. 创建 src/models/ctr/my_model.py (或 multitask/)
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 实现模型

    def forward(self, x):
        # 实现前向传播

# 2. 创建配置 configs/models/my_model.yaml
# 3. 在实验配置中使用
```

### 添加新分析方法
```python
# 1. 创建 src/analysis/my_analysis.py
class MyAnalyzer:
    @staticmethod
    def analyze(data):
        # 实现分析逻辑
        pass

# 2. 编写测试 tests/test_my_analysis.py
```

## 📦 运行输出结构

每次运行生成 `runs/{TIMESTAMP}_{EXPERIMENT}/`：

```
runs/2026-01-01_12-34-56_avazu_deepfm/
├── config.yaml                  # 使用的配置副本
├── run_meta.json               # git/env/seed/hash信息
├── dataset_manifest.json       # 数据集契约
├── feature_map.json            # 特征映射契约
├── metrics.json                # 所有指标结果
├── artifacts/
│   ├── train.parquet          # 训练集特征
│   ├── valid.parquet          # 验证集特征
│   ├── feature_stats.json     # 特征统计
│   └── auction_stream.parquet # 仿真流（可选）
├── checkpoints/
│   └── model.pt               # 最佳模型
├── curves/
│   ├── train_loss.png
│   ├── auc.png
│   └── calibration_curve.png
├── tables/
│   ├── metrics_summary.csv
│   └── significance_test.csv
└── notes.md                    # 实验报告
```

## 🔐 最佳实践

### ✅ 应该做
- ✓ 继承基类实现功能
- ✓ 定义类型注解
- ✓ 添加docstring
- ✓ 编写单元测试
- ✓ 使用配置文件控制行为
- ✓ 返回契约定义的数据类型
- ✓ 记录重要中间结果

### ❌ 不应该做
- ✗ 硬编码参数
- ✗ 直接修改输入数据
- ✗ 忽视类型安全
- ✗ 跳过测试
- ✗ 导入循环依赖
- ✗ 在模块间共享全局状态

## 📞 常见问题

### Q: 如何复现某个实验？
A: 使用 `run_meta.json` 中的git commit、config hash和seed。

### Q: 如何对比两个模型？
A: 使用 `src.cli.evaluate` 的 `--baseline-run-id` 参数进行显著性检验。

### Q: 如何添加新的评估指标？
A: 在 `src/trainers/evaluator_offline.py` 或 `src/analysis/` 中添加计算函数。

### Q: 数据缓存在哪里？
A: `data/processed/` 中的parquet文件。

### Q: 如何调试训练过程？
A: 查看 `runs/{run_id}/` 中的日志文件或使用PyCharm/VSCode调试器。

## 🎓 学习路径

1. **理解架构** - 阅读 README.md 和 SETUP_COMPLETE.md
2. **运行示例** - 执行 `pytest tests/` 理解测试用例
3. **修改配置** - 尝试在 `configs/` 中改参数并观察效果
4. **实现适配器** - 为新数据集实现 `BaseAdapter`
5. **添加模型** - 实现新的模型类
6. **自定义分析** - 在 `src/analysis/` 中添加新方法

## 📚 相关文件

- **README.md** - 完整文档
- **CONTRIBUTING.md** - 贡献指南
- **SETUP_COMPLETE.md** - 搭建总结
- **pyproject.toml** - 项目元数据
- **contracts/examples/** - 数据示例（待填充）

---

**记住：契约先行，测试驱动，配置控制！** 🚀

