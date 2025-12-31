# 贡献指南

感谢对本项目的兴趣！本指南将帮助您有效地贡献代码。

## 开发流程

### 1. 克隆仓库

```bash
git clone <repo_url>
cd ctr_llm_creative
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. 安装开发依赖

```bash
pip install -e ".[dev,notebook]"
```

### 4. 代码风格

我们使用以下工具确保代码质量：

```bash
# 自动格式化
black src/ tests/

# Import排序
isort src/ tests/

# Linting
flake8 src/ tests/

# 类型检查
mypy src/
```

### 5. 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_feature_map_consistency.py -v

# 生成覆盖率报告
pytest tests/ --cov=src/ --cov-report=html
```

## 添加新功能

### 添加数据集适配器

1. 在 `src/data/adapters/` 中创建新文件
2. 继承 `BaseAdapter`
3. 实现 `load_split()` 和 `get_features()`
4. 在 `src/core/registry.py` 中注册
5. 编写测试用例

示例：
```python
from src.data.adapters.base import BaseAdapter
from contracts import DatasetManifest

class MyDatasetAdapter(BaseAdapter):
    def load_split(self, split: str):
        # 实现数据加载逻辑
        pass
    
    def get_features(self):
        # 返回特征定义
        pass
```

### 添加新模型

1. 在 `src/models/` 的相应目录中创建新文件
2. 实现 `torch.nn.Module` 或自定义基类
3. 在 `src/core/registry.py` 中注册
4. 添加模型配置到 `configs/models/`

### 添加新的分析方法

1. 在 `src/analysis/` 中创建新文件
2. 实现相应的分析类/函数
3. 编写测试用例
4. 更新 README 和文档

## 代码组织原则

### 模块划分
- **contracts/**: 只定义数据结构，不含业务逻辑
- **core/**: 平台级通用功能
- **data/**: 数据加载和处理
- **models/**: 模型定义
- **trainers/**: 训练循环
- **sim/**: 仿真逻辑
- **analysis/**: 分析方法

### 命名规范
- Python文件: `snake_case`
- 类: `PascalCase`
- 函数/变量: `snake_case`
- 常量: `UPPER_SNAKE_CASE`

### 文档
- 每个模块添加模块文档字符串
- 每个公共函数/类添加docstring
- 复杂逻辑添加inline注释

## 提交规范

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type
- `feat`: 新功能
- `fix`: 修复bug
- `refactor`: 代码重构
- `style`: 代码风格（不影响逻辑）
- `test`: 测试相关
- `docs`: 文档
- `chore`: 构建、依赖等

### 例子

```
feat(data): add iPinYou adapter

Implement BaseAdapter for iPinYou dataset.
Support loading auction stream and generating AuctionStream contracts.

Closes #42
```

## Pull Request 流程

1. Fork本仓库
2. 创建特性分支: `git checkout -b feature/my-feature`
3. 提交更改: `git commit -am 'Add my feature'`
4. 推送到分支: `git push origin feature/my-feature`
5. 提交 Pull Request

### PR检查清单

- [ ] 代码按照风格指南格式化
- [ ] 添加了相关测试
- [ ] 测试通过（100% 覆盖率优先）
- [ ] 更新了相关文档
- [ ] 提交信息清晰

## 性能指南

### 数据加载
- 使用 Parquet 格式（压缩效率好）
- 实现分批加载，避免内存溢出

### 模型训练
- 使用梯度累积处理大批次
- 实现模型量化和蒸馏

### 仿真
- 并行处理多个拍卖
- 缓存计算结果

## 文档更新

- 对代码修改或新功能同时更新 README
- 对API更改更新 API文档
- 在 CHANGELOG 中记录重要变更

## 问题报告

提交issue时请包括：
1. 问题描述
2. 复现步骤
3. 预期行为
4. 实际行为
5. 环境信息（Python版本、依赖版本等）

## 许可

贡献代码即表示您同意在 MIT License 下发布您的代码。

---

感谢您的贡献！ 🎉
