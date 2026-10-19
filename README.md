# CUDA Kernels for LLM Inference

个人实现的 LLM 推理常用 CUDA 算子集合，包含访存优化、性能分析与 profiling 记录。用于深入理解 GPU 执行模型与底层算子优化。

## 算子列表与优化点

| 算子 | 源文件 | 优化技术 / 说明 |
|---|---|---|
| Decode Attention | `test_decodeattn.cu` | 基于 KV Cache 的单 token 解码，访存合并 |
| FlashAttention | `test_flashattention.cu` | 分块计算，记录并分析了 ncu profiling 数据（见下文） |
| GEMM | `test_GEMM.cu` | shared memory tiling、register tiling、bank conflict 规避，包含多种尺寸的边界测试（2~128），验证非 64 倍数下的正确性。 |
| LayerNorm | `test_layernorm.cu` | warp-level 归约，shared memory 通信 |
| Reduce | `test_reduce.cu` | warp shuffle 优化，避免 shared memory bank conflict |
| Scan (Prefix Sum) | `test_scan.cu` | 基于 __shfl_up_sync 的 Warp-level 局部前缀和，配合 Shared Memory 完成 Block 级归约 |
| Softmax | `test_softmax.cu` | 按行归约，防止数值溢出（减去 max） |
| Matrix Transpose | `test_transpose.cu` | shared memory padding 规避 bank conflict |

## FlashAttention 性能分析 (ncu)

在实现最简版 FlashAttention 的基础上，使用 Nsight Compute (ncu) 进行了性能分析，定位到如下瓶颈：

**4×4 计算块配置（基线）**：
- **低占用率**：Theoretical Occupancy 仅为 8.3%。受限于 Shared Memory 用量（41KB），每个 SM 只能容纳 2 个 Block（每 Block 64 线程，即 2 个 Warp），实际 Occupancy 仅 4.45%。
- **Bank Conflict**：73% 的 Shared Memory 访问为多余的 Wavefront（uncoalesced shared accesses）。

**2×2 计算块尝试**：
- 为降低 Shared Memory 用量、提升 Occupancy，尝试将每线程计算块从 4×4 调整为 2×2。
- **结果**：ncu 显示 Bank Conflict 占比反而上升至 81%。由于未能定位到具体的 Source Line，最终回退至基线配置。
- **反思**：后续应先使用 ncu 的 *L1 Wavefronts Shared Excessive* 表格定位到具体代码行，再决定是修改 Shared Memory Padding 还是调整线程映射，而非盲目修改块尺寸。

## 编译与运行

环境要求：CUDA Toolkit (nvcc)
*(注：请根据你的实际情况修改编译命令)*

```bash
# 以编译 GEMM 为例
nvcc test_GEMM.cu -o test_GEMM
./test_GEMM