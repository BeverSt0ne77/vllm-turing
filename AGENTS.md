# Agent Instructions for vllm-turing

> 本仓库是 `vllm-project/vllm` v0.29.0 的个人分支，用于 **2 × RTX 2080 Ti 22G + NVLink** 的定向优化。
> 不使用上游 CI（`.github/`、`.buildkite/` 已移除），也不走上游 PR/合并流程。除非明确要求，不要新增 workflow 或 CI 配置。

## 1. 硬件约束（改动前必读）

目标 GPU 是 Turing 架构，compute capability **SM 7.5**，缺少 Ampere 及以后的部分特性：

- **不要引入 bf16 依赖**：SM 7.5 不支持，需用 float16（`--dtype=half`）。
- **不要引入 FP8 依赖**：需 SM ≥ 8.9，本机不支持。
- **不要默认 FlashAttention / FlashInfer**：均需 SM ≥ 8.0；注意力走 **Triton 后端**（`TRITON_ATTN`）。
- 双卡通过 NVLink 互联，分布式推理默认走 **TP=2**。
- 本机必须**从源码编译**：不要用 `VLLM_USE_PRECOMPILED=1`，预编译产物缺少 SM 7.5 的 kernel。

修改涉及架构判断的代码时，确认新路径在 SM 7.5 上不会因能力门控而不可用。

## 2. 开发环境

- **不要用系统 `python3` 或裸 `pip`**。统一走 `uv` 和 `.venv/bin/python`。

```bash
# 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate

# 源码编译安装（按机器调整 MAX_JOBS，过高易 OOM）
MAX_JOBS=$(nproc) uv pip install -e . --no-build-isolation

# 安装 pre-commit 钩子
uv pip install -r requirements/lint.txt
pre-commit install
```

C/C++ 或 CUDA 改动请参考
[增量编译流程](docs/contributing/incremental_build.md)。

## 3. 测试

```bash
# 安装测试依赖
uv pip install -r requirements/test/cuda.in

# 运行单个测试文件
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

- 先设计再写：明确模块用途、I/O 契约、要防的失败，以及最低成本能覆盖它的层级（unit > integration > e2e）。
- 优先复用已有的测试文件、`conftest.py` fixture 与 helper，没有合适位置才新建文件。
- 通过公开 API 断言可观察行为；不稳定（flaky）的测试比没有测试更糟。
- kernel 性能实验放 `benchmarks/kernels/`，不要塞进 `tests/`。
- 涉及模型输出/精度/服务的改动，跑一次 `tests/evals/` 或 `vllm bench` 并附结果。

## 4. 代码风格

- 匹配现有代码风格。
- 少写注释：优先让代码自解释，注释和 docstring 简短直接。
- Python 行宽上限 88，不确定就用 pre-commit 检查。
- 用 [Google 风格 docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)（`Args:`/`Returns:`/`Raises:`），不要用 `:param:`/`:return:` 这类 Sphinx 字段。

### 运行 linter

```bash
# 对暂存文件跑全部钩子
pre-commit run

# 对全部文件
pre-commit run --all-files

# 单个钩子
pre-commit run ruff-check --all-files

# mypy（manual 阶段，与上游 CI 配置一致）
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

## 5. 提交信息

使用 `Co-authored-by:` 等 trailer 标注 AI 协助，并按需添加 `Signed-off-by:`：

```text
提交标题

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

## 6. 安全相关

安全审查请先读 [`SECURITY.md`](SECURITY.md)、
[`docs/usage/security.md`](docs/usage/security.md) 和
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)。
