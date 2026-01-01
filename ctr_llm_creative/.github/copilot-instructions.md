**Purpose**
- **简要**: 帮助 AI 编码代理快速在本仓库中上手，聚焦项目架构、关键工作流、约定与示例命令。

**Big Picture（总体架构）**
- **职责分离**: 配置驱动的训练/仿真/评估流水线。训练、仿真和评估各自位于 src/cli、src/sim、src/analysis 或 src/cli 路径下。
- **配置中心**: 使用 `configs/`（Hydra 风格的层级覆盖），基础参数在 [configs/_base/base.yaml](configs/_base/base.yaml)。实验在 `configs/experiments/` 下定义。
- **插件式注册**: 全局插件注册器位于 [src/core/registry.py](src/core/registry.py)，用于数据适配器、模型和模拟器的注册与查找。

**关键组件与数据流（快速参考）**
- **Run 管理**: 运行目录与产物由 `RunManager` 管理（[src/core/run_manager.py](src/core/run_manager.py)）。运行目录结构包含 `artifacts/`, `checkpoints/`, `curves/`, `tables/`。
- **训练流程**: CLI 入口为 [src/cli/train.py](src/cli/train.py)，训练循环骨架在 [src/trainers/trainer.py](src/trainers/trainer.py)。模型、优化器与数据加载被注入到 `Trainer` 中。
- **仿真/评估**: 仿真器原型在 [src/sim/auction_simulator.py](src/sim/auction_simulator.py)，合约/数据结构定义在 `contracts/` 包中（查阅 `contracts/auction_stream.py` 等）。

**常用开发/运行命令（举例）**
- **快速跑整条流水线**: 脚本示例 `scripts/run_experiment.sh`：

  ```bash
  bash scripts/run_experiment.sh
  ```

- **手工启动训练（示例）**:

  ```bash
  python -m src.cli.train --config configs/experiments/avazu_infra_deepfm.yaml
  ```

- **准备数据**（脚本化入口）:

  ```bash
  python scripts/prepare_data.py --dataset avazu --data-dir data/
  ```

- **运行测试**:

  ```bash
  pytest -q
  ```

**项目约定与模式（agent 应遵循）**
- **配置优先**: 不要在代码中硬编码实验参数；优先寻找 `configs/` 下的 YAML 并沿用 Hydra 风格覆盖（参见 base.yaml）。
- **使用 Registry 注册/检索**: 新模型/模拟器应通过 `models`、`simulators`、`dataset_adapters` 全局 registries 注册（见 [src/core/registry.py](src/core/registry.py)）。代理生成代码时应优先采用注册/解耦模式。
- **产物写入位置**: 使用 `RunManager` 来创建 run 目录并保管 config 与模型文件，避免在仓库根目录散落产物（见 [src/core/run_manager.py](src/core/run_manager.py)）。
- **接口契约**: 仿真/拍卖相关的类型与流定义在 `contracts/`，任何改动需要同时更新契约实现与调用方。

**示例参考（小片段）**
- 从配置复制到 run:

  - See [src/core/run_manager.py](src/core/run_manager.py) `save_config()`。

- 注册模型示例:

  - See [src/core/registry.py](src/core/registry.py) `models.register()` / `models.get()`。

**调试与本地开发提示**
- 若要调试训练流程，可在 [src/cli/train.py](src/cli/train.py) 中临时调用 `train()` 并传入小规模 config（`data.cache_data: true`、`training.epochs: 1`）以快速迭代。
- 仿真相关改动请先创建单元/集成测试，参考 `tests/` 下现有用例风格；仿真会依赖 `contracts` 中的记录结构。

**限制与注意事项（不得假设的内容）**
- 项目使用 Hydra 风格配置层，但并非所有 CLI 入口都已实现完整 Hydra API（部分 CLI使用 argparse）。先检查目标脚本（例如 [src/cli/train.py](src/cli/train.py)）再决定如何传参。
- 若需新增长期运行任务（GPU/分布式），先查 `pyproject.toml` 与 `requirements.txt` 确认依赖并在 `configs/` 中添加设备/分布式参数。

**EDA 模块（Evidence-Driven Exploratory Data Analysis）**
- **位置**: [src/eda/avazu_evidence_eda.py](src/eda/avazu_evidence_eda.py)
- **目标**: 为 FeatureMap 设计与模型选择提供量化证据（Decision → Metric → Artifact → Code）
- **设计约束**: 流式处理（chunksize-based），禁止全量加载；多遍扫描支持；所有输出落到 `data/interim/{dataset}/`
- **关键输出**:
  - `eda/overview.json`, `schema.json`: 数据概览与异常检测
  - `eda/topk/*.csv`, `eda/time/*.csv`, `eda/drift/*.csv`: 分析产物
  - `featuremap/featuremap_spec.yml`: FeatureMap 设计决策（供 src/data/adapters 使用）
  - `model_plan/model_plan.yml`: 模型架构建议（供 src/cli/train.py 参考）
  - `reports/featuremap_evidence.md`, `reports/model_structure_evidence.md`: 决策链证明
- **运行示例**: `python src/eda/avazu_evidence_eda.py --input_csv data/raw/avazu/train.csv --out_root data/interim/avazu --chunksize 500000 --verbose`
- **文档**: [src/eda/INTEGRATION_GUIDE.md](src/eda/INTEGRATION_GUIDE.md), [src/eda/QUICK_REFERENCE.md](src/eda/QUICK_REFERENCE.md), [src/eda/IMPLEMENTATION_PROGRESS.md](src/eda/IMPLEMENTATION_PROGRESS.md)

**如需更多信息，请检查**
- 项目入口说明: [README.md](README.md)
- 配置示例: [configs/_base/base.yaml](configs/_base/base.yaml)
- 训练入口: [src/cli/train.py](src/cli/train.py)
- 运行管理: [src/core/run_manager.py](src/core/run_manager.py)
- 注册器模式: [src/core/registry.py](src/core/registry.py)
- 仿真: [src/sim/auction_simulator.py](src/sim/auction_simulator.py)
- **EDA 集成**: [src/eda/INTEGRATION_GUIDE.md](src/eda/INTEGRATION_GUIDE.md) (新增模块说明)

如果哪部分不清楚或需要把某个文件/函数列为“优先学习项”，告诉我我会把该部分扩写成更具体的 agent 指令。
