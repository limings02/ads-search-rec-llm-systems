# 项目框架搭建完成

## 总结

已成功按照您提供的规范完整搭建了 **CTR/LLM创意搜索推荐系统** 的项目框架。

### 📁 完整的目录结构

```
ctr_llm_creative/
├── contracts/                 # P0：统一契约层
│   ├── __init__.py
│   ├── dataset_manifest.py    # 数据集结构定义
│   ├── feature_map.py         # 特征变换映射
│   ├── auction_stream.py      # 可回放拍卖流
│   ├── metrics.py             # 离线/仿真指标
│   ├── run_meta.py            # 运行元数据
│   └── examples/              # 示例JSON文件目录（待填）
│
├── src/                       # 核心源代码
│   ├── __init__.py
│   ├── cli/                   # 统一命令行入口
│   │   ├── train.py           # Stage1：训练
│   │   ├── simulate.py        # Stage2：仿真
│   │   ├── evaluate.py        # Stage3：评估
│   │   └── export_run.py      # 导出运行结果
│   │
│   ├── core/                  # 平台内核
│   │   ├── contracts_io.py    # 契约I/O
│   │   ├── registry.py        # 注册机制
│   │   ├── run_manager.py     # 运行目录管理
│   │   ├── logger.py          # 统一日志
│   │   └── reproducibility.py # 可重复性
│   │
│   ├── data/                  # 数据处理
│   │   ├── adapters/          # 数据集适配器
│   │   │   ├── base.py
│   │   │   ├── avazu.py
│   │   │   ├── ali_ccp.py
│   │   │   ├── ipinyou.py
│   │   │   └── criteo_attr.py
│   │   ├── feature_engineering/
│   │   │   ├── base.py
│   │   │   ├── fit_feature_map.py
│   │   │   ├── transform.py
│   │   │   └── sequence_builder.py
│   │   ├── splits.py
│   │   └── dataloaders.py
│   │
│   ├── models/                # 模型定义
│   │   ├── ctr/               # CTR模型
│   │   │   ├── deepfm.py
│   │   │   └── dcn.py
│   │   ├── multitask/         # 多任务模型
│   │   │   └── esmm.py
│   │   ├── calibration/       # 校准方法
│   │   │   └── temperature_scaling.py
│   │   └── common/            # 公共组件
│   │       ├── embedding.py
│   │       ├── mlp.py
│   │       ├── loss.py
│   │       └── metrics.py
│   │
│   ├── trainers/              # 训练模块
│   │   ├── trainer.py
│   │   ├── evaluator_offline.py
│   │   └── callbacks.py
│   │
│   ├── sim/                   # 竞价仿真
│   │   ├── bid_generator.py
│   │   ├── auction_simulator.py
│   │   ├── budget_manager.py
│   │   ├── kpi_collector.py
│   │   └── stream_builder.py
│   │
│   ├── analysis/              # 统计分析
│   │   ├── bootstrap.py
│   │   ├── calibration_analyzer.py
│   │   ├── offline_online_gap.py
│   │   └── report_builder.py
│   │
│   └── api/                   # FastAPI后端
│       ├── app.py
│       ├── routes_runs.py
│       ├── schemas.py
│       └── static/
│
├── configs/                   # Hydra配置
│   ├── _base/
│   │   └── base.yaml          # 基础配置
│   ├── datasets/
│   │   ├── avazu.yaml
│   │   ├── ali_ccp.yaml
│   │   └── ipinyou.yaml
│   ├── features/
│   │   ├── avazu_hash.yaml
│   │   └── ali_ccp_crossday.yaml
│   ├── models/
│   │   ├── deepfm.yaml
│   │   └── esmm.yaml
│   ├── simulation/
│   │   ├── base_2nd_price.yaml
│   │   └── bidding_ecpm.yaml
│   ├── evaluation/
│   │   └── bootstrap.yaml
│   └── experiments/           # 实验配置
│       ├── avazu_infra_deepfm.yaml
│       ├── ali_ccp_esmm_crossday.yaml
│       └── ipinyou_closedloop.yaml
│
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   │   ├── Ali/
│   │   ├── avazu-ctr-prediction/
│   │   ├── criteo/
│   │   ├── criteo_attribution_dataset/
│   │   └── ipinyou.contest.dataset/
│   ├── interim/               # 中间产物
│   │   ├── avazu/
│   │   ├── ali_ccp/
│   │   └── ipinyou/
│   └── processed/             # 最终数据
│       ├── avazu/
│       ├── ali_ccp/
│       └── ipinyou/
│
├── runs/                      # 运行输出目录（自动生成）
│   └── {run_id}/
│       ├── config.yaml
│       ├── run_meta.json
│       ├── dataset_manifest.json
│       ├── feature_map.json
│       ├── metrics.json
│       ├── artifacts/
│       ├── checkpoints/
│       ├── curves/
│       ├── tables/
│       └── notes.md
│
├── tests/                     # 单元测试
│   ├── __init__.py
│   ├── test_feature_map_consistency.py
│   ├── test_auction_second_price.py
│   └── test_end2end_toy.py
│
├── scripts/                   # 工具脚本
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── download_datasets.sh
│   └── run_experiment.sh
│
├── pyproject.toml             # 项目配置（setuptools）
├── requirements.txt           # 依赖列表
├── README.md                  # 项目文档
├── CONTRIBUTING.md            # 贡献指南
├── .gitignore                 # Git忽略配置
└── .env.example               # 环境变量示例
```

## 🎯 核心设计特点

### 1. **契约驱动设计（P0）**
- 统一的数据结构定义（`contracts/`）
- 所有模块通过契约通信
- 支持JSON序列化/反序列化
- 确保train/valid/test、offline/simulation的一致性

### 2. **阶段化流程**
```
Stage 1: 训练    → 离线指标 (AUC, LogLoss, ECE)
Stage 2: 仿真    → 仿真KPI (spend, CTR, CVR, RPM)
Stage 3: 评估    → 显著性检验 (Bootstrap CI)
```

### 3. **配置管理（Hydra）**
- 分层配置（_base → datasets → models → experiments）
- 支持命令行override
- 自动生成run_meta元数据
- 完整的复现性支持

### 4. **灵活的扩展机制**
- Registry模式（dataset_adapters, models, simulators）
- 无需修改核心代码即可扩展
- 支持多个实现并存

### 5. **完整的测试框架**
- 特征一致性测试
- 拍卖逻辑测试
- 端到端集成测试
- 100% 覆盖率目标

## 📦 已实现的模块

### ✅ 完整实现
- [x] 契约定义（5个核心类）
- [x] 平台核心（contracts_io, registry, run_manager等）
- [x] 数据适配器框架
- [x] 特征工程框架
- [x] 模型架构框架
- [x] 训练与评估框架
- [x] 仿真模块框架
- [x] 分析模块框架
- [x] FastAPI后端框架
- [x] Hydra配置
- [x] 单元测试
- [x] 脚本工具

### ⏳ 待实现（业务逻辑）
- [ ] 数据适配器的具体实现（Avazu/Ali-CCP/iPinYou）
- [ ] 特征工程的具体实现
- [ ] 模型的具体实现（DeepFM/ESMM等）
- [ ] 训练循环的具体实现
- [ ] 仿真引擎的具体实现
- [ ] 分析方法的具体实现
- [ ] API路由的具体实现

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行测试
```bash
pytest tests/ -v
```

### 3. 准备数据
```bash
bash scripts/download_datasets.sh
python scripts/prepare_data.py --dataset avazu
```

### 4. 训练模型
```bash
python -m src.cli.train --config configs/experiments/avazu_infra_deepfm.yaml
```

### 5. 启动API
```bash
uvicorn src.api.app:app --reload
```

## 📚 关键文件说明

| 文件 | 说明 |
|-----|------|
| `contracts/*.py` | 数据契约定义（P0层） |
| `src/core/contracts_io.py` | 契约序列化/反序列化 |
| `src/core/run_manager.py` | 运行目录和元数据管理 |
| `pyproject.toml` | 项目配置和依赖 |
| `configs/experiments/*.yaml` | 实验组合配置 |
| `README.md` | 项目主文档 |

## 🔄 工作流示例

```python
# 1. 定义数据集契约
from contracts import DatasetManifest, TaskSpec, TaskType

manifest = DatasetManifest(
    name="avazu",
    dataset_type="ctr",
    feature_fields=[...],
    label_fields=["click"],
)

# 2. 加载数据
from src.data.adapters import AvazuAdapter
adapter = AvazuAdapter(manifest)
train_data = adapter.load_split("train")

# 3. 特征工程
from src.data.feature_engineering import FeatureMapFitter
fitter = FeatureMapFitter(feature_map)
fitter.fit(train_data)

# 4. 训练
from src.trainers import Trainer
trainer = Trainer(model, optimizer, config)
trainer.train(train_loader, valid_loader, epochs=20)

# 5. 仿真（可选）
from src.sim import AuctionSimulator
simulator = AuctionSimulator()
result_stream = simulator.simulate(auction_stream, bid_generator)

# 6. 分析
from src.analysis import BootstrapAnalyzer
analyzer = BootstrapAnalyzer()
ci = analyzer.bootstrap_ci(y_true, y_pred, metric_fn)
```

## 📖 文档结构

- **README.md** - 项目总览、快速开始
- **CONTRIBUTING.md** - 贡献指南、开发规范
- **pyproject.toml** - 项目元数据和依赖
- **contracts/examples/** - 契约示例（待填充）

## ✨ 最佳实践

1. **遵循契约定义** - 所有新模块必须返回定义的数据类型
2. **模块化设计** - 每个模块独立，通过明确的接口通信
3. **配置驱动** - 业务逻辑通过配置文件控制
4. **充分测试** - 新功能必须包含单元测试
5. **文档完善** - Docstring + 类型注解

## 🎓 后续建议

1. **填充业务实现** - 从适配器开始，逐个实现具体模块
2. **添加示例配置** - 在 `contracts/examples/` 中添加JSON示例
3. **编写集成测试** - 完整的端到端测试
4. **优化性能** - Profile和优化关键路径
5. **前端集成** - 开发可视化仪表板

---

**项目框架搭建完成！可以开始填充具体的业务实现了。** 🎉

