# vllm-turing

> 面向 **2 × RTX 2080 Ti 22G + NVLink** 定向优化的 vLLM 个人分支

本仓库 fork 自 [vllm-project/vllm](https://github.com/vllm-project/vllm) **v0.29.0**。目标是让 vLLM 在自己的机器上跑得尽可能快、尽可能稳；除硬件适配外尽量与上游保持一致，方便后续跟随上游 rebase。

## 目标硬件

| 项目 | 值 |
| --- | --- |
| GPU | 2 × RTX 2080 Ti 22G（改装显存版） |
| 架构 | NVIDIA Turing，compute capability **SM 7.5** |
| 互联 | NVLink（2-way），双卡可跑张量并行 `TP=2` |

## 硬件限制与应对

Turing 架构缺少 Ampere 及以后的部分特性，使用时有几个硬性约束：

| 特性 | 上游要求 | 本机情况 | 应对 |
| --- | --- | --- | --- |
| bfloat16 | SM ≥ 8.0 | 不支持 | 改用 float16：`--dtype=half` |
| FP8 | SM ≥ 8.9 | 不支持 | 改用 INT8/INT4 量化；注意部分量化 kernel 同样要求 SM80+ |
| FlashAttention | SM ≥ 8.0 | 不可用 | 使用 Triton 注意力后端：`--attention-backend TRITON_ATTN` |
| FlashInfer | SM ≥ 8.0（SM75 暂被禁用） | 不可用 | 同上，走 Triton |
| NVLink | — | 可用 | `--tensor-parallel-size 2` |

> vLLM 启动时会按设备能力自动筛选可用后端，在 2080 Ti 上会落到 Triton 等可用后端。显式指定 `TRITON_ATTN` 可避免反复探测与误选。

## 环境要求

- Python 3.10 – 3.14（推荐 3.12）
- CUDA 12.x（由 PyTorch 提供）
- PyTorch 2.13.0
- **从源码编译**：上游预编译 wheel 不含 SM 7.5 的完整 kernel，不能直接用于 Turing

## 从源码安装

遵循上游约定，Python 命令统一走 `uv`，不要用系统 `python3` / 裸 `pip`：

```bash
# 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate

# 本地编译安装（MAX_JOBS 按 CPU 核心数与内存调整，过高易 OOM）
MAX_JOBS=$(nproc) uv pip install -e . --no-build-isolation
```

> 提示：不要设置 `VLLM_USE_PRECOMPILED=1`，它使用上游预编译产物，在 Turing 上可能缺少对应架构的 kernel。

## 双卡运行示例

启动 OpenAI 兼容 API 服务：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --dtype half \
  --attention-backend TRITON_ATTN \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92
```

离线批量推理：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,
    dtype="half",
    attention_backend="TRITON_ATTN",
)

outputs = llm.generate(
    ["你好，请简单介绍一下你自己。"],
    SamplingParams(max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

> 单卡 22G 显存有限，7B 模型 FP16 权重约需 15G；双卡 `TP=2` 可留出充足的 KV cache 空间。更大的模型可叠加量化使用。

## 与上游的差异

- 已移除整个 `.github/`（工作流、issue 模板、CODEOWNERS、mergify、dependabot 等）与 `.buildkite/` 流水线，本分支不使用上游 CI。
- 后续会加入针对 SM 7.5 与 NVLink 的定向优化，详见提交记录。

## 上游资源

- 文档：<https://docs.vllm.ai>
- 论文（PagedAttention）：<https://arxiv.org/abs/2309.06180>
- 上游仓库：<https://github.com/vllm-project/vllm>

## 引用

如果本分支对你的研究有帮助，请引用 vLLM 原始论文：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
