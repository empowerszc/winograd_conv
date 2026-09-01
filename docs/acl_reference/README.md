# ACL Winograd 参考文档

> 本目录包含从 oneDNN 源码树复制的 ACL（Arm Compute Library）Winograd 卷积实现分析文档。
>
> 这些文档是 `winograd_conv` 项目的**参考实现指南**，用于理解 ACL 的算法原理、汇编优化策略，
> 指导后续进一步优化（特别是 SVE/SME 汇编变换、arm_gemm GEMM 内核等）。
>
> A1/A2/A3 落地后（2026-08-10 复测），`winograd_conv` 在 t16 微基准上已 **8/9 case 快于 oneDNN**（快 1.68~3.54x，仅 Case 2 慢 1.13x）。
> ⚠️ **已过时（2026-09-01 更新）**：最新用 arm_gemm 后端 + 59 形状对照（见 `../docs/onednn_comparison.md` §五，已闭环）：
> **我们的 F(4,4) 全面碾压 oneDNN 的 wino:acl 1.15-9.5x**（同算法 PK，e2e_wino + benchdnn_wino 两列互相验证）。
> 但 ACL 的手写汇编变换 + arm_gemm JIT 内核仍是参考优化目标（尤其小形状 direct conv 路径）。
>
> 详见 `../PERFORMANCE_ANALYSIS.md` 的差距分析和下一步优化建议。

## 文档列表

| 文件 | 内容 | 大小 |
|------|------|------|
| `acl_wino_neon_intrinsics_annotated.md` | NEON C++ intrinsics 变换逐行注释（输入/输出/权重） | 17 KB |
| `acl_wino_neon_asm_annotated.md` | NEON 内联汇编（a64_fp32_6x6.cpp）逐行注释 | 17 KB |
| `acl_wino_sve_asm_annotated.md` | SVE 内联汇编（sve_fp32_6x6.cpp）逐行注释 + 指令速查 | 11 KB |
| `acl_wino_sme_asm_annotated.md` | SME 内联汇编（sme_fp32_mla_6x6 + sme_fp32_mopa_4x4_3x3）逐行注释 | 18 KB |
| `acl_wino_implementation_details.md` | ACL Winograd 实现深度剖析（矩阵推导、数据布局、调度） | 47 KB |
| `acl_wino_transform_kernels_explained.md` | 变换 kernel 源码分析（10 章，含公式/数值示例/数据布局） | 83 KB |
| `acl_23.11_wino_transform_kernels_explained.md` | ACL 23.11 版变换 kernel 文档 | 22 KB |
| `acl_23.11_vs_53.1.0_wino_analysis.md` | ACL 23.11 vs 53.1.0 版本对比（SVE kernel 差异 + 汇编入门） | 36 KB |

## 使用场景

1. **优化权重变换**：参考 `acl_wino_neon_intrinsics_annotated.md` 中 ACL 的手写公式（直接展开 G 矩阵行，跳过 Ww 中间缓冲区）
2. **优化变换实现**：参考 `acl_wino_sve_asm_annotated.md` 中 ACL 的 SVE 汇编（完全展开、无分支、指令调度）
3. **替换 GEMM**：参考 `acl_wino_implementation_details.md` 中 arm_gemm 的内核选择机制
4. **理解 SME FMOPA**：参考 `acl_wino_sme_asm_annotated.md` 中 Kronecker 积 + FMOPA 的 157 条指令逐行注释
