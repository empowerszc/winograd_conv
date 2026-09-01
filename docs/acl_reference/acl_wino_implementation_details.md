# oneDNN acl:wino 路线实现深度解析

> **适用范围**：oneDNN 3.12.1（`src/cpu/aarch64`）+ Arm Compute Library 53.1.0（`D:\300Code\ComputeLibrary-53.1.0`）。
> **目标机器**：华为鲲鹏 920F（920 专业版），AArch64 / Armv9-A + **SVE-512** + **SME**（Scalable Matrix Extension），16 NUMA × 38 核 = 608 核 + 已编译 ACL（`DNNL_AARCH64_USE_ACL=ON`）。
> **目的**：把 `acl_wino_convolution_fwd_t`（oneDNN 分发表里前向 f32 的 **第 1 号候选**，verbose 名 `wino:acl`）从"被选中"到"算出结果"的每一步实现细节讲透，便于演示与分享。
>
> 本文所有矩阵系数、函数签名、内存布局均 **逐行取自源码** 并标注文件:行号。

---

## 目录

- [1. 定位：acl:wino 是什么、何时被选中](#1-定位aclwino-是什么何时被选中)
- [2. 端到端数据流总览](#2-端到端数据流总览)
- [3. oneDNN 侧：选择条件与参数翻译](#3-onednn-侧选择条件与参数翻译)
- [4. ACL 侧：CpuWinogradConv2d 总体结构](#4-acl-侧cpuwinogradconv2d-总体结构)
- [5. Winograd 数学原理](#5-winograd-数学原理)
- [6. ACL 变换实现选择（get_implementation）](#6-acl-变换实现选择get_implementation)
- [7. 变换矩阵详解（F(2,2,3,3)，源码级）](#7-变换矩阵详解f2233-源码级)
- [8. GEMM 步骤详解](#8-gemm-步骤详解)
- [9. 并行与线程模型](#9-并行与线程模型)
- [10. 内存管理与 workspace](#10-内存管理与-workspace)
- [11. 融合：bias + activation + sum](#11-融合bias--activation--sum)
- [12. 目标机器（SVE-512 + SME，608 核）与实际运行 case 分析](#12-目标机器sve-512--sme608-核与实际运行-case-分析)
- [13. 性能特性与局限](#13-性能特性与局限)
- [14. 关键文件索引](#14-关键文件索引)

---

## 1. 定位：acl:wino 是什么、何时被选中

在 oneDNN 前向 f32 分发表（`cpu_convolution_list.cpp:150`）里，`acl_wino_convolution_fwd_t` 是 **第 1 号候选**（仅 ACL 编译开启时存在）。它的 `DECLARE_COMMON_PD_T` 名字是 `"wino:acl"`（`acl_winograd_convolution.hpp:73`）。

它委托 ACL 的 `arm_compute::NEWinogradConvolutionLayer`（`acl_winograd_convolution.hpp:32`），后者内部又委托 `arm_compute::cpu::CpuWinogradConv2d`（`src/cpu/operators/CpuWinogradConv2d.{h,cpp}`）。真正的 Winograd 变换与 GEMM 在 ACL 的汇编内核（`src/core/NEON/kernels/convolution/winograd/`）里完成。

**被选中的条件**（见 `acl_winograd_convolution.hpp:75` 的 `init()` + `acl_convolution_utils.cpp:287` 的 `init_conf_wino()`）：

1. `is_fwd()`；数据类型全 f32 或全 f16。
2. `alg_kind ∈ {convolution_auto, convolution_winograd}`。
3. `!has_zero_dim_memory()`。
4. **`DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL`**（仅 OpenMP/TBB 可用，threadpool 不行——因为 Winograd 内部用 mutex + resource_mapper，不支持并发多线程访问）。
5. `init_conf_wino()` 形状约束：
   - auto 模式下，若 `ih>112 || iw>112 || ic<64 || oc<64 || 线程数>28` 之一成立 → **主动回退**（让位给 GEMM 卷积）。
   - 仅 **stride=1×1**、**pad≤1**、**无 dilation**、**2D（ndims==4）**、**无 groups**。
   - 通过 ACL `NEWinogradConvolutionLayer::validate()`。

> 也就是说：典型的 **3×3 stride=1 小图（如 ResNet 早期 56×56、28×28 层）、ic/oc≥64、线程≤28** 的卷积会命中 acl:wino；否则回退给 `brgemm_conv<sve_512>`。

---

## 2. 端到端数据流总览

Winograd 卷积把"空间窗口卷积"拆成三步：**输入变换 → Winograd 域逐元素乘（实为 batched GEMM）→ 输出变换**。

```
                 oneDNN 用户张量 (NHWC/NCHW, 指针)
                              │  import_memory（零拷贝，挂指针）
                              ▼
        ┌──────────────────────────────────────────────────┐
        │  ACL NEWinogradConvolutionLayer (CpuWinogradConv2d) │
        │                                                  │
        │  [prepare 阶段，一次性]                            │
        │   weights OHWI ─permute→ HWIO ─weight_transform→  │
        │       transformed_weights (Winograd 域)           │
        │                                                  │
        │  [run 阶段，每次前向]                              │
        │   ① 若 NCHW: permute src→NHWC                    │
        │   ② input_transform:  src(空间) → U(Winograd域)   │  ← B^T d B
        │   ③ GEMM:  C[m,n] = U[m,k] × V[k,n]  (batched)   │  ← n_multis 个 GEMM
        │   ④ output_transform: M(Winograd域) → dst(空间)   │  ← A^T M A (+bias +ReLU)
        │   ⑤ 若 NCHW: permute output→NCHW                 │
        └──────────────────────────────────────────────────┘
                              │  post_ops.execute()（oneDNN 侧，sum/binary等）
                              ▼
                       oneDNN 用户 dst
```

**核心三步对应 ACL 的三个 kernel**（`NEWinogradConvolutionLayer.h:43-48` 注释）：
- `CpuWinogradConv2dTransformInputKernel`（输入变换）
- `CpuGemmAssemblyDispatch`（Winograd 域 GEMM）
- `CpuWinogradConv2dTransformOutputKernel`（输出变换，含 bias+激活）
- 外加最多三次 `CPPPermute`（权重、输入、输出，仅 NCHW 时需要输入/输出 permute）

---

## 3. oneDNN 侧：选择条件与参数翻译

### 3.1 `pd_t::init()`（`acl_winograd_convolution.hpp:75-110`）

```cpp
status_t init(engine_t *engine) {
    // ① 数据类型与属性粗筛
    const bool is_fp16_ok = expect_data_types(f16,f16,f16,f16,undef)
            && attr()->has_default_values(skip_mask_t::post_ops, f16);
    const bool is_fp32_ok = expect_data_types(f32,f32,f32,f32,undef)
            && attr()->has_default_values(skip_mask_t::post_ops, f32);
    bool ok = is_fwd()
            && one_of(desc()->alg_kind, convolution_auto, convolution_winograd)
            && one_of(true, is_fp16_ok, is_fp32_ok)
            && !has_zero_dim_memory();
    ok = ok && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL;  // ② 禁 threadpool
    if (!ok) return status::unimplemented;
    // ③ 形状深层校验（含 auto 回退、stride/pad/dilation/2D/groups 约束）
    CHECK(acl_convolution_utils::init_conf_wino(acp_, src_md_, weights_md_,
            dst_md_, bias_md_, *desc(), *attr()));
    set_default_alg_kind(alg_kind::convolution_winograd);
    // ④ 翻译 post_ops → act_info（仅 RELU/BOUNDED_RELU 能融进输出变换）
    CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_, acp_.act_info));
    acp_.use_dst_acc_for_sum = post_ops.has_sum();   // ⑤ sum 后处理
    ...book scratchpad for sum accumulator...
    return status::success;
}
```

### 3.2 `init_conf_wino()`（`acl_convolution_utils.cpp:287-334`）

```cpp
status_t init_conf_wino(...) {
    // ① auto 模式主动回退条件
    if (one_of(true, src_md.dims[2] > 112,  // ih
                     src_md.dims[3] > 112,  // iw
                     src_md.dims[1] < 64,   // ic
                     dst_md.dims[1] < 64,   // oc
                     dnnl_get_max_threads() > 28)
            && cd.alg_kind == alg_kind::convolution_auto) {
        return status::unimplemented;   // 让位给 GEMM 卷积
    }
    // ② 通用参数翻译
    acp.alg_winograd = true;
    CHECK(acl_init_conf(acp, src_md, weights_md, dst_md, bias_md, cd, attr));
    // ③ 形状硬约束
    const bool shape_ok
            = (acp.padstride_info.stride() == std::pair<uint,uint>{1,1})
            && (acp.padstride_info.pad().first <= 1)   // 左/右 pad
            && (acp.padstride_info.pad().second <= 1)  // 上/下 pad
            && (acp.dilation_info == arm_compute::Size2D(1, 1));  // 无 dilation
    ACL_CHECK_SUPPORT(!shape_ok, "shape not supported by winograd kernels");
    // ④ ACL 自校验
    ACL_CHECK_VALID(arm_compute::NEWinogradConvolutionLayer::validate(
        &acp.src_tensor_info, &acp.wei_tensor_info,
        acp.with_bias ? &acp.bia_tensor_info : nullptr,
        &acp.dst_tensor_info, acp.padstride_info, acp.act_info, true));
    return status::success;
}
```

> 注意第 ④ 步的 `true` 是 `enable_fast_math`。oneDNN **始终以 fast_math=true 调用 ACL Winograd**（见 3.3 的 configure），因为部分配置（F(4×4,5×5)、F(2×2,5×5) 等）只在 fast_math 下可用。

### 3.3 `acl_init_conf()` 的参数翻译（`acl_convolution_utils.cpp:36-285`）

把 oneDNN 描述符翻译成 ACL 对象：

| oneDNN 侧 | ACL 侧 | 说明 |
|-----------|--------|------|
| `cd.strides` | `acp.padstride_info`（`PadStrideInfo`） | stride + 上下左右 pad + FLOOR rounding |
| `cd.dilates` | `acp.dilation_info`（`Size2D`） | oneDNN dilation 加 1 = ACL dilation（:94-98） |
| `cd.bias_desc` | `acp.with_bias` | bias 是否存在 |
| src/wei/dst memory_desc | `acp.{src,wei,dst,bia}_tensor_info`（`TensorInfo`） | NHWC 或 NCHW，shape 按布局重排（:170-203） |
| `attr.fpmath_.mode_` | `acp.fast_math` | bf16/any → ACL fast_math（**wino 路径在 :213 提前 return，不走 fixed-format 权重**） |

关键点：Winograd 路径在 `acl_init_conf` 中提前返回（`acl_convolution_utils.cpp:206-214`），**不走 ACL 的 fixed-format 权重优化**（`WeightFormat::ANY` / `has_opt_impl` 那套），因为 Winograd 有自己的权重变换流程。

### 3.4 `configure()` 与 `execute_forward()`

**configure**（`acl_winograd_convolution.hpp:44-52`，在 `create_resource` 时调用一次）：
```cpp
acl_wino_obj_->conv.configure(
    &src_tensor, &wei_tensor,
    acp.with_bias ? &bia_tensor : nullptr,
    &dst_tensor,
    acp.padstride_info, acp.act_info,
    true);  // enable_fast_math，同时开启 5x5/7x7 支持
```

**execute_forward**（`acl_winograd_convolution.cpp:25-39` + `acl_convolution_utils.hpp:190-232`）：
```cpp
std::lock_guard<std::mutex> _lock {this->mtx};   // ① 全局锁
// ② import_memory：把 oneDNN 用户指针挂到 ACL Tensor（零拷贝）
acl_wino_obj.src_tensor.allocator()->import_memory(src_base);
acl_wino_obj.wei_tensor.allocator()->import_memory(wei_base);
...dst, bias...
// ③ run workspace：把 oneDNN scratchpad 挂到 ACL aux tensor
// ④ ACL 执行
acl_wino_obj.conv.run();
// ⑤ 后处理：sum/binary 等 post_ops
pd->post_ops.execute(ctx, dst);
```

> ① 的 mutex 是因为 `resource_mapper` 不支持并发多线程访问——这也是 threadpool 运行时被禁用的根因。

---

## 4. ACL 侧：CpuWinogradConv2d 总体结构

`CpuWinogradConv2d`（`src/cpu/operators/CpuWinogradConv2d.h:43`）是 ACL 的 CPU Winograd 算子，持有：

```cpp
// CpuWinogradConv2d.h:116-137
std::unique_ptr<CpuGemm>         _gemm_function;            // Winograd 域 GEMM
std::unique_ptr<ICPPKernel>      _transform_input_kernel;   // 输入变换 kernel
std::unique_ptr<ICPPKernel>      _transform_output_kernel;  // 输出变换 kernel
std::unique_ptr<CpuPermute>      _permute_input/_output/_weights;  // NCHW↔NHWC
arm_conv::winograd::WinogradImpl _winograd_impl;            // 选中的变换三元组 + GEMM 参数
arm_conv::ConvolutionArgs        _conv_args;
TensorInfo _winograd_transformed_input/_output/_weights;    // Winograd 域张量
TensorInfo _input_workspace/_output_workspace;              // 变换的工作空间
```

### 4.1 `configure()`（`CpuWinogradConv2d.cpp:174-322`）

1. **选变换**：`get_winograd_kernel_implementation()`（:192）→ `arm_conv::winograd::get_implementation<float>()`，根据 kernel 尺寸、数据类型、fast_math、CPUInfo 选出 input/weight/output 三个变换（见第 6 节）。
2. **算 GEMM 维度**（:223-227）：从 `_winograd_impl.gemm_args` 取
   - `_Msize` = 输出 tile 总数（M）
   - `_Ksize` = 输入通道数（K）
   - `_Nsize` = 输出通道数（N）
   - `_nmulti` = Winograd 域矩阵数（= n_multis，见第 8 节）
   - `_nbatches` = batch
3. **构造 Winograd 域 TensorInfo**（:230-256）：A(变换后输入)、B(变换后权重)、D(变换后输出)，strides 取自 `winograd_spec`（`input_ld_row`/`input_ld_matrix`/`input_ld_batch` 等）。
4. **permute 配置**（:258-278）：NCHW 时配输入/输出 permute；权重总要从 OHWI→HWIO permute。
5. **配 input transform kernel / GEMM / output transform kernel**（:281-290）。
6. **配 activation**（:293-297）：仅当激活非 RELU/BOUNDED_RELU（不能融进输出变换）时单独跑 `CpuActivation`。
7. **算 aux_mem workspace**（:299-320）：每个 slot 的 size/alignment/生命周期。

### 4.2 `prepare()`（`CpuWinogradConv2d.cpp:426-475`，一次性）

```cpp
if (!_is_prepared) {
    // ① permute weights OHWI → HWIO
    _permute_weights->run(...);
    // ② Winograd 权重变换：HWIO 空间权重 → Winograd 域
    _winograd_impl.weight_transform->execute(*_conv_args,
        permuted_weights_ptr, ..., win_wght_transf_ptr,
        _winograd_impl.winograd_spec, 0, 1);  // 单线程
    // ③ GEMM 权重预准备（如 arm_gemm 内部 reorder）
    _gemm_function->prepare(gemm_pack);
    _is_prepared = 1;
}
```

> 权重变换只在 `prepare()` 做 **一次**（`MemoryLifetime::Prepare`），之后每次 `run()` 复用变换后的权重——这是 Winograd 的关键优势之一：把"核窗口累加"的代价在权重侧一次性付清。

### 4.3 `run()`（`CpuWinogradConv2d.cpp:359-424`，每次前向）

```cpp
void run(ITensorPack &tensors) {
    prepare(tensors);
    // ① NCHW → NHWC（若需要）
    if (is_nchw) _permute_input->run({{ACL_SRC, src}, {ACL_DST, input_nhwc}});

    // ② 输入变换：src(NHWC) → winograd_input_transformed
    NEScheduler::get().schedule_op(_transform_input_kernel.get(), DimX, win, transform_input_pack);

    // ③ GEMM：winograd_input × winograd_weights → winograd_output
    _gemm_function->run(gemm_pack);   // 跑 n_multis 个 batched GEMM

    // ④ 输出变换：winograd_output → dst(NHWC)（+bias +ReLU）
    NEScheduler::get().schedule_op(_transform_output_kernel.get(), DimX, win, transform_output_pack);

    // ⑤ NHWC → NCHW（若需要）+ 未融合激活（若有）
    if (is_nchw) _permute_output->run(...);
    if (_run_activation) _activation_func->run(...);
}
```

> 注释 :399 "Run 16 GEMMs in multiple threads"——这正是 F(2,2,3,3) 的 16 个 Winograd 域 GEMM。

---

## 5. Winograd 数学原理

### 5.1 核心公式

Winograd 最小滤波算法把一个 `r×r` 核在 `m×m` 输出 tile 上的卷积写成：

```
Y = A^T · [ (G · g · G^T) ⊙ (B^T · d · B) ] · A
         \___________/     \_____________/
          变换后权重 V         变换后输入 U
         (一次性,prepare)    (每次前向)
                              \________________________/
                               ⊙ 改成 batched GEMM: M = U·V
```

其中：
- `d` = 输入 tile（大小 `(m+r-1) × (m+r-1)`）
- `g` = 卷积核（`r × r`）
- `B`、`G`、`A` 是固定的变换矩阵（仅依赖 m, r）
- `⊙` 在 Winograd 域是逐元素乘，但因为要对 **所有 tile、所有通道** 做，整体组织成 **batched GEMM**

### 5.2 乘法量对比（以 F(2,2,3,3) 为例）

| | 直接卷积 | Winograd F(2,2,3,3) |
|---|---------|----------------------|
| 每 2×2 输出 tile 的乘法 | 3×3×3×3 = ... 实际每输出像素 9 次 | 4×4 = 16 次（覆盖 2×2=4 像素） |
| 每输出像素等效乘法 | 9 | 16/4 = **4** |
| 加法开销 | 0 | 额外的 B/G/A 变换加法 |

**乘法量降到约 4/9 ≈ 44%**，加法增加但乘法是瓶颈，净收益为正——前提是形状"小核+小图+unit stride"。

### 5.3 输出 tile 大小的权衡

ACL 支持多种 F(m,r) 配置（见第 6 节）。**m 越大**：
- ✅ 每 tile 覆盖更多输出像素，**tile 总数更少**，GEMM 的 M 更小但每个 GEMM 更"宽"
- ✅ 权重复用更充分
- ❌ Winograd 域矩阵数 `(m+r-1)²` 更多（GEMM 数更多），变换加法开销更大
- ❌ 对小特征图，tile 填充浪费多

所以 ACL 的选择策略是 **"贪心选满足约束的最大输出 tile"**（第 6 节）。

### 5.4 tile 划分（`input_transform.hpp:93-98`）

```cpp
const auto tile_stride_rows = max(1u, m_input_rows - kernel_rows + 1);  // = m（输出 tile 高）
const auto n_tile_rows = iceildiv(output_shape.rows, m_input_rows - kernel_rows + 1);
const auto n_tile_cols = iceildiv(output_shape.cols, m_input_cols - kernel_cols + 1);
```

即：输入 tile 大小 = `m+r-1`，tile 步长 = `m`（输出 tile 大小），tile 数 = `ceil(oh/m) × ceil(ow/m)`。边界用 padding 处理（`TransformUnpadded` 把有效部分拷进工作空间补零，`input_transform.hpp:303-356`）。

```
F(2,2,3,3) tile 划分示意（输入 tile 4×4，输出 tile 2×2，步长 2）：

  输入特征图 (补零后)
  ┌───┬───┬───┬───┬───┐
  │ 0 │ 0 │ 0 │ 0 │ 0 │  ← padding 行/列
  ├───┼───┼───┼───┼───┤
  │ 0 │■■■│■■■│■■■│   │
  │ 0 │■d0■│■d1■│   │     d0 = 4×4 tile → 变换 → U0
  ├───┼───┼───┼───┤     d1 = 4×4 tile → 变换 → U1
  │ 0 │■■■│■■■│■■■│   │     (两个 tile 在空间上重叠 2 列——Winograd 允许重叠)
  │ 0 │■d2■│■d3■│   │
  ├───┼───┼───┼───┤
  │ 0 │ ...       │
  └───┴───┴───┴───┴───┘
  每个 4×4 输入 tile 产生 2×2 输出 tile
  输出: ceil(oh/2) × ceil(ow/2) 个 tile
```

---

## 6. ACL 变换实现选择（`get_implementation`）

### 6.1 三类变换注册表

ACL 把变换分成三类，每类有一组按 **优先级排序** 的实现（fp32）：

**输入变换**（`input_transforms_fp32.cpp:51-67`，tile 大小即 `(m+r-1)`）：

| 顺序 | 实现 | tile | 约束 | 适用 |
|------|------|------|------|------|
| 1 | `sme_fp32_mla_6x6` | 6×6 | RequiresSME | **本文目标机命中（SME 可用）** |
| 2 | `sve_fp32_6x6` | 6×6 | RequiresSVE | 有 SVE 无 SME 的核（对比机，非本文目标） |
| 3 | `a64_fp32_6x6` | 6×6 | 无 | 通用 AArch64（NEON） |
| 4 | `arm_fp32_4x4` | 4×4 | 无 | F(2,2,3,3) |
| 5 | `arm_fp32_1x8`（含转置 8×1） | 1×8 | 无 | 1D 变体 |

**输出变换**（`output_transforms_fp32.cpp:50-66`，命名 `F(out_h×out_w, kern_h×kern_w)`）：

| 顺序 | 实现 | F(m,r) | 约束 |
|------|------|--------|------|
| 1 | `sme_fp32_mopa_4x4_3x3` | F(4,4,3,3) | RequiresSME（**本文目标机命中**） |
| 2 | `arm_fp32_4x4_3x3` | F(4,4,3,3) | **LargerShape**（输入 > 输出 tile） |
| 3 | `arm_fp32_2x2_3x3` | F(2,2,3,3) | 无 |
| 4 | `arm_fp32_2x2_5x5` | F(2,2,5,5) | 无 |
| 5-10 | `arm_fp32_1x6_1x3` / `1x4_1x5` / `1x2_1x7`（含转置） | 1D | 无 |

**权重变换**（`weight_transforms_fp32.cpp:49-64`，命名 `F(out, kern)`，变换后 tile = `(m+r-1)`）：

| 顺序 | 实现 | F(m,r) | 变换后 tile |
|------|------|--------|-------------|
| 1 | `arm_fp32_4x4_3x3` | F(4,4,3,3) | 6×6 |
| 2 | `arm_fp32_2x2_3x3` | F(2,2,3,3) | 4×4 |
| 3 | `arm_fp32_2x2_5x5` | F(2,2,5,5) | 6×6 |
| 4-9 | `cpp_fp32_1x6_1x3` / `1x4_1x5` / `1x2_1x7`（含转置） | 1D | 1×8 |

### 6.2 匹配算法（`winograd_implementations.hpp:236-339`）

```
get_implementation(dest, ci, conv_args, max_threads, fast_mode, cfg, gemm_cfg):

  1. 收集所有满足约束的 weight_transforms（kernel 尺寸匹配）
  2. 收集所有满足约束的 input_transforms（target tile 匹配）
  3. 收集所有满足约束的 output_transforms（kernel 尺寸 + 输出 tile 匹配）
  4. ★ 贪心：按 output_transforms 注册顺序（最大输出 tile 在前）遍历
     for each output_transform:
       for each weight_transform:
         if weight.transformed_tile == output.input_tile:   // 变换后 tile 必须一致
           for each input_transform:
             if input.tile == output.input_tile:            // 输入 tile 必须一致
               选中 (output, weight, input) 三元组 → success, break
  5. 算 GEMM 参数（见第 8 节）
```

**关键点**：
- output_transforms 列表里 **大输出 tile 排前面**（SME 4×4 → 4×4 → 2×2 …），所以默认贪心选 **最大的可用输出 tile**。
- `LargerShape` 约束（`winograd_implementations.hpp:74`）：F(4,4,3,3) 的非 SME 输出变换 `arm_fp32_4x4_3x3` 要求 `input.rows > 4 && input.cols > 4`，即 **大特征图才用 4×4**；小特征图（如 4×4 以下）自动落到 F(2,2,3,3)。注意 SME 版 `sme_fp32_mopa_4x4_3x3` **无 LargerShape 约束**，所以小图也优先选它。
- 在 **本文目标机（SVE-512 + SME）** 上：SME 变换可用且排最前 → 3×3 卷积选 `sme_fp32_mopa_4x4_3x3`（输出，F(4,4,3,3)）+ `sme_fp32_mla_6x6`（输入，6×6 tile）+ `arm_fp32_4x4_3x3`（权重，变换到 6×6）。SME 的 `mopa`（矩阵外积累加）指令让输入/输出变换与 GEMM 都能借助 2D tile 寄存器，吞吐远高于纯 SVE/NEON。
- 对比无 SME 的机器（如某些 SVE-512 服务器、A64FX 等）：SME 不可用 → 退到 `sve_fp32_6x6`（输入）+ `arm_fp32_4x4_3x3`（输出，需 LargerShape）。

### 6.3 支持的 2D F 配置矩阵（fp32）

| kernel \ 输出tile | 2×2 | 4×4 |
|-------------------|-----|-----|
| 3×3 | F(2,2,3,3) | F(4,4,3,3)（需 LargerShape 或 SME） |
| 5×5 | F(2,2,5,5) | — |
| 1×3 / 1×5 / 1×7 | 1D 变体 | — |

> F(4,4,5,5)（tile 8×8，64 GEMM）等需要 fast_math，oneDNN 始终传 true，但 ACL fp32 注册表里未直接列出 4×4_5x5，主要由 F(2,2,5,5) 覆盖 5×5。

---

## 7. 变换矩阵详解（F(2,2,3,3)，源码级）

下面三组矩阵 **逐行取自 ACL 源码**，与 Lavin & Fast 论文的标准 F(2,2,3,3) 矩阵一致。

### 7.1 权重变换矩阵 G（`weight_transforms/arm_fp32_2x2_3x3.cpp:60-83`）

变换 `V = G · g · G^T`，代码先算 `Ww = G · w`（:60-69），再算 `V = Ww · G^T`（:72-83）：

```cpp
Ww[0][j] = w[0][j];                                    // 1,   0,   0
Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);          // 0.5, 0.5, 0.5
Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);          // 0.5,-0.5, 0.5
Ww[3][j] = w[2][j];                                    // 0,   0,   1
```

故：

```
        ┌ 1     0     0   ┐
   G =  │ 0.5   0.5   0.5 │   (4×3, 把 3×3 权重 → 4×4 Winograd 域)
        │ 0.5  -0.5   0.5 │
        └ 0     0     1   ┘
```

权重变换用 NEON `vld1q_f32`/`vmulq_n_f32` 一次处理 4 个输出通道（:43-96），再降到 2 通道、1 通道的标量路径（:98-195）。

### 7.2 输入变换矩阵 B^T（`input_transforms/arm_fp32_4x4.cpp:99-128`）

变换 `U = B^T · d · B`，代码先算 `XTx = B^T · x`（:99-112），再算 `U = XTx · B`（:115-128）：

```cpp
XTx[0][j] = x[0][j] - x[2][j];    //  1,  0, -1,  0
XTx[1][j] = x[1][j] + x[2][j];    //  0,  1,   1,  0
XTx[2][j] = x[2][j] - x[1][j];    //  0, -1,   1,  0
XTx[3][j] = x[1][j] - x[3][j];    //  0,  1,   0, -1
```

故：

```
          ┌ 1  0 -1  0 ┐
   B^T =  │ 0  1   1  0 │   (4×4, 把 4×4 输入 tile → 4×4 Winograd 域)
          │ 0 -1   1  0 │
          └ 0  1   0 -1 ┘
```

输入变换驱动 `TransformUnpadded`（`input_transform.hpp:286-377`）处理 padding：若 tile 触及边界，先把有效部分拷进 per-thread 工作空间补零，再跑变换 kernel（:303-356）。

### 7.3 输出变换矩阵 A^T（`output_transforms/arm_fp32_2x2_3x3.cpp:64-81`）

变换 `f = A^T · M · A`（M 是 GEMM 出来的 Winograd 域结果），代码先算 `FZ = F · A`（:64-71），再算 `f = A^T · FZ`（:74-81）：

```cpp
FZ[i][0] = F[i][0] + F[i][1] + F[i][2];   //  1,  1,  1,  0
FZ[i][1] = F[i][1] - F[i][2] - F[i][3];   //  0, -1, -1, -1
```

故（A 是 4×2，A^T 是 2×4）：

```
          ┌ 1  1  1  0 ┐
   A^T =  │           │   (2×4, 把 4×4 Winograd 域 → 2×2 输出)
          └ 0 -1 -1 -1 ┘
```

### 7.4 端到端数值示例（F(2,2,3,3)，单通道单 tile）

设输入 tile `d`（4×4，含 padding）与核 `g`（3×3）：

```
   d = ┌ 1 2 3 4 ┐       g = ┌ 1 1 1 ┐
       │ 5 6 7 8 │           │ 1 1 1 │
       │ 9 0 1 2 │           └ 1 1 1 ┘
       └ 3 4 5 6 ┘

  ① U = B^T d B      → 4×4 Winograd 输入
  ② V = G g G^T      → 4×4 Winograd 权重（prepare 阶段算好）
  ③ M = U ⊙ V        → 4×4（实为 GEMM: M[tile,oc] = U[tile,ic]·V[oc,ic]，对每个 Winograd 矩阵）
  ④ f = A^T M A      → 2×2 输出（= 直接卷积 2×2 输出位置的结果）

  f 的 4 个值 = 直接卷积在对应 2×2 输出位置的值，但只用了 4 次乘法（Winograd 域逐元素乘）
  而非 9 次/像素。
```

> 实际实现里，③ 不是逐元素 ⊙ 而是把它组织成 **n_multis 个 batched GEMM**（第 8 节），让所有 tile/所有通道一次性算完，复用 GEMM 库的高吞吐。

---

## 8. GEMM 步骤详解

### 8.1 GEMM 维度（`winograd_implementations.hpp:299-320`）

```cpp
const auto n_output_row_tiles = iceildiv(output_shape.rows, output_transform.get_output_rows());  // ceil(oh/m)
const auto n_output_col_tiles = iceildiv(output_shape.cols, output_transform.get_output_cols());  // ceil(ow/m)
const auto n_output_patches  = n_output_row_tiles * n_output_col_tiles;   // M = tile 总数

const int n_multis = input_transform.get_input_rows() * input_transform.get_input_cols();
// n_multis = (m+r-1)²  ← Winograd 域矩阵数 = GEMM 个数

dest.gemm_args.reset(new GemmArgs(
    ci,
    n_output_patches,         // M  = tile 总数
    n_output_channels,        // N  = OC
    n_input_channels,         // K  = IC
    1,                        // K-sections
    n_batches,                // # Batches
    n_multis,                 // # multi GEMMs
    false,                    // Indirect input
    {},                       // No activation（激活在输出变换里做）
    max_threads, fast_mode, gemm_cfg));
```

| 参数 | 含义 | F(2,2,3,3) 示例 |
|------|------|-----------------|
| M | 输出 tile 总数 = `ceil(oh/2)×ceil(ow/2)` | 56×56→28×28=784 |
| N | 输出通道数 OC | 64 |
| K | 输入通道数 IC | 64 |
| n_multis | Winograd 域矩阵数 = `(m+r-1)²` | (2+3-1)²=**16** |
| #batches | batch size | 1 |

> **所以 F(2,2,3,3) 跑 16 个 GEMM**，每个 `M×N×K` = `tiles × OC × IC`。注释 `CpuWinogradConv2d.cpp:399` "Run 16 GEMMs" 正是此意。
> F(4,4,3,3) 跑 `(4+3-1)²=36` 个 GEMM，但 M 更小（tiles = ceil(oh/4)×ceil(ow/4)）。

### 8.2 Winograd 域内存布局（`winograd_implementations.hpp:322-336`）

```cpp
ws.weight_ld_row      = iroundup(n_output_channels, 4u);          // N 按 4 对齐
ws.weight_ld_matrix   = n_input_channels * ws.weight_ld_row;      // K * ld_row
ws.weight_matrix_size = n_multis * ws.weight_ld_matrix * sizeof(T);

ws.input_ld_row       = iroundup(n_input_channels, 4u);           // K 按 4 对齐
ws.input_ld_matrix    = iroundup(n_output_patches, 4u) * ws.input_ld_row;  // M*K
ws.input_ld_batch     = n_multis * ws.input_ld_matrix;            // 跨 n_multis
ws.input_matrix_size  = n_batches * ws.input_ld_batch * sizeof(T);

ws.output_ld_row      = ws.weight_ld_row;                         // = N 对齐
ws.output_ld_matrix   = n_output_patches * ws.output_ld_row;     // M*N
ws.output_ld_batch    = n_multis * ws.output_ld_matrix;
ws.output_matrix_size = n_batches * ws.output_ld_batch * sizeof(T);
```

**4D 张量布局**（`CpuWinogradConv2d.cpp:230-256`）：
- A（变换后输入）：shape `[K, M, n_batches, n_gemms]`，strides `[dt_size, input_ld_row*dt_size, ...]`
- B（变换后权重）：shape `[N, K, n_gemms]`
- D（变换后输出）：shape `[N, M, n_batches, n_gemms]`

`n_gemms` 维（最外层）就是 n_multis——GEMM 沿这一维 batched。

### 8.3 GEMM 本体

`_gemm_function` 是 `CpuGemm`（`src/cpu/operators/CpuGemm.h`），内部委托 `arm_gemm` 库。在本文目标机（SVE-512 + SME）上，arm_gemm 会优先选 **SME GEMM 内核**（利用 2D tile 外积），其次 SVE-512 内核。oneDNN 调 `_gemm_function->run(gemm_pack)`（`CpuWinogradConv2d.cpp:405`），它跨 n_multis + batches 多线程跑所有小 GEMM。

> GEMM 不带 activation（`GemmArgs` 传 `{}` 无激活），因为激活融在输出变换里。

---

## 9. 并行与线程模型

### 9.1 变换 kernel 的细粒度线程

`run()` 里（`CpuWinogradConv2d.cpp:370-371`）：
```cpp
win.set(Window::DimX, Window::Dimension(0, nthreads, 1));  // 每线程一个迭代
NEScheduler::get().schedule_op(_transform_input_kernel.get(), Window::DimX, win, pack);
```

注释 :370 "The Winograd transform implementation does fine-grain threading inside the transforms"——变换 kernel **自己内部分 tile 给线程**（见 `input_transform.hpp:106` 的 `for tile_i = thread_id; tile_i < n_tile_rows; tile_i += n_threads`）。oneDNN 只传 `thread_id` 和 `nthreads`，不在外层按 tile 切。

### 9.2 GEMM 的多线程

`_gemm_function->run()` 内部由 `arm_gemm` 按 M/N 维多线程跑。`GemmArgs` 传 `max_threads`。

### 9.3 为什么 oneDNN 加 mutex

`execute_forward` 全程 `std::lock_guard<std::mutex>`（`acl_winograd_convolution.cpp:29`）。原因（注释 :27-28）：`resource_mapper` 不支持并发多线程访问。所以 Winograd 路径 **整体串行**（一个卷积实例同时只能一个线程跑 run），这是 threadpool 运行时被禁用的根因——threadpool 模式下多线程并发跑同一 primitive 会冲突。

```
线程模型对比：
  acl:wino:        mutex 串行 + 变换/GEMM 内部各自多线程
  brgemm_conv:     无锁，按 mb*nb_oc*nb_ow 切分给线程并发
  → 多线程高并发场景 wino 受限，这也是 auto 模式"线程>28 回退"的原因
```

---

## 10. 内存管理与 workspace

### 10.1 AuxTensorIdx（`CpuWinogradConv2d.h:104-115`）

```cpp
enum AuxTensorIdx {
    TransformedInput = 7,     // 变换后输入 U
    TransformedOutput,        // 变换后输出 M（=GEMM 结果）
    WorkspaceIO,              // 输入/输出变换的 per-thread 工作空间（取较大者，因为时间不重叠）
    TransformedWeights,       // 变换后权重 V（Prepare 生命周期）
    PermutedWeights,          // permute 后的 HWIO 权重（Prepare 生命周期）
    Count,
    PermutedInput = TransformedOutput,   // 复用：NCHW permute 与 TransformedOutput 时间不重叠
    PermutedOutput = TransformedInput,  // 复用：与 TransformedInput 时间不重叠
};
```

### 10.2 内存生命周期（`CpuWinogradConv2d.cpp:306-320`）

| slot | 内容 | 生命周期 | size |
|------|------|----------|------|
| TransformedInput | Winograd 域输入 U | Temporary | `input_matrix_size_bytes`（align 64） |
| TransformedOutput | Winograd 域输出 M | Temporary | `output_matrix_size_bytes`（align 64） |
| WorkspaceIO | 变换 per-thread 工作空间 | Temporary | `max(input_ws, output_ws)`（**复用，因为输入/输出变换时间不重叠**） |
| PermutedWeights | HWIO 权重 | **Prepare** | `weights_hwio.total_size()` |
| TransformedWeights | Winograd 域权重 V | **Prepare** | `weight_matrix_size_bytes`（align 64） |
| PermutedInput/Output | NCHW↔NHWC 中转 | Temporary | 复用 TransformedOutput/Input |

`WorkspaceIO` 取 `max(input_ws, output_ws)` 而非两者之和——因为输入变换（②）在输出变换（④）之前完成，两者的工作空间时间不重叠，可复用同一块内存（注释 :305）。

### 10.3 与 oneDNN scratchpad 的对接

oneDNN 在 `init_scratchpad`（`acl_convolution_utils.hpp:82-108`）把 ACL `conv.workspace()` 的每个 aux_mem slot 映射到 oneDNN scratchpad key：
```cpp
const auto aux_mem_req = conv.workspace();
for (const auto &key : conv_keys) {
    if (aux_mem_req[id].size > 0)
        scratchpad.book(key.second, aux_mem_req[id].size, 1,
                        aux_mem_req[id].alignment, aux_mem_req[id].alignment);
}
```
`run()` 时（`acl_convolution_utils.hpp:164-177`）把 oneDNN scratchpad 指针 `import_memory` 进 ACL 临时 Tensor，挂进 `ITensorPack`。**全程零拷贝**，ACL 直接用 oneDNN scratchpad 当 workspace。

---

## 11. 融合：bias + activation + sum

### 11.1 输出变换里的 fused bias + ReLU

输出变换 kernel（`output_transforms/arm_fp32_2x2_3x3.cpp:84-104`）在写回前：
```cpp
// 加 bias
b = bptr ? vld1q_f32(bptr) : vdupq_n_f32(0.0f);
// 加 bias + clamp（fused ReLU / Bounded ReLU）
const auto y = vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b),
                                   vdupq_n_f32(output_max)),
                         vdupq_n_f32(output_min));
vst1q_f32(outptr + ..., y);
```

`output_min`/`output_max` 来自 `ActivationLayerInfo`：
- 普通 ReLU：min=0, max=+INF
- Bounded ReLU：min=0, max=给定上界
- 无激活：min=-INF, max=+INF（clamp 退化为恒等）

所以 **bias + (bounded) ReLU 融在输出变换里**，无额外 pass。

### 11.2 act_info 翻译

oneDNN `post_ops.init(engine, attr_.post_ops_, dst_md_, acp_.act_info)`（`acl_winograd_convolution.hpp:98`）把 oneDNN 的 eltwise post-op 翻译成 `arm_compute::ActivationLayerInfo`。`fuse_function_supported`（`CpuWinogradConv2d.cpp:139-143`）只认 RELU/BOUNDED_RELU：
```cpp
return act_info.activation() == RELU || act_info.activation() == BOUNDED_RELU;
```
能融合的 → 进 `act_info`，融在输出变换；不能融合的（如 GELU/Swish）→ `_run_activation=true`，输出变换后单独跑 `CpuActivation`（`CpuWinogradConv2d.cpp:293-297, 419-423`）。

### 11.3 sum 后处理

若 post_ops 有 sum（残差连接），oneDNN 设 `use_dst_acc_for_sum=true`（`acl_winograd_convolution.hpp:100`），让 ACL 把卷积结果写进一块 scratchpad accumulator（`key_generic_acc`），而非直接写 dst；之后 `post_ops.execute()` 再把 accumulator 累加进 dst（含 sum 的 scale）。这样 sum 不被卷积覆盖。

---

## 12. 目标机器（SVE-512 + SME，608 核）与实际运行 case 分析

### 12.0 机器规格与对 acl:wino 的两层判定

| 项 | 值 | 对 acl:wino 的影响 |
|----|----|---------------------|
| 芯片 | **华为鲲鹏 920F（920 专业版）**，自研核，Armv9-A | 本文目标机 |
| ISA | SVE-512 + **SME** | SME 变换（`sme_fp32_mla_6x6` 输入、`sme_fp32_mopa_4x4_3x3` 输出）**可用且排最前**，3×3 卷积优先选 F(4,4,3,3) 的 SME 版 |
| 拓扑 | 16 NUMA × 38 核 = **608 核** | `dnnl_get_max_threads()` 远大于 28 → **auto 模式下所有 case 触发"线程>28"回退** |
| ACL | 已编译 | `acl_wino` 条目存在 |

oneDNN `init_conf_wino` 有 **两类拦截**，理解它们是分析每个 case 的前提：

1. **auto 回退**（`acl_convolution_utils.cpp:295-302`，仅 `alg_kind==convolution_auto` 时生效）：
   `ih>112 || iw>112 || ic<64 || oc<64 || threads>28` 任一为真 → `unimplemented`。
2. **形状硬约束**（:309-317，无论 auto/winograd 都生效）：
   `stride==1×1 && pad<=1 && 无dilation && 2D && 无groups`，否则 `ACL_CHECK_SUPPORT` 失败。

> **本机关键结论**：608 核使 `threads>28` 恒真。因此在默认 `convolution_auto` 模式下，**全部 11 个 case 都会被 auto 回退**，落到 `brgemm_conv<sve_512>`。只有以下两种情形 acl:wino 才会真正生效：
> - 用户显式设 `alg_kind = convolution_winograd`（绕过 auto 回退，但仍受形状硬约束）；
> - 或把线程数压到 ≤28（如 `OMP_NUM_THREADS=28`，但单 NUMA 有 38 核，仍 >28，故需进一步限制）。
>
> ⚠️ **2026-09-01 实测修正**：上述结论基于默认 608 线程。实际 benchmark 用 `DNNL_NUM_THREADS=16` +
> `OMP_THREAD_LIMIT=16`（nthr=16 ≤ 28），**auto 模式不回退**，59 shape 中 40 个选 `wino:acl`、
> 19 个选 `brgconv:sve_512`（见 `docs/onednn_comparison.md` §五）。且我们的 F(4,4) 全面碾压
> wino:acl 1.15-9.5x（同算法 PK）。

### 12.1 SME 可用时的变换三元组（3×3 卷积）

本机有 SME，对 **3×3、stride=1** 卷积，ACL `get_implementation` 选：
- output：`sme_fp32_mopa_4x4_3x3`（F(4,4,3,3)，RequiresSME，注册表第 1，无 LargerShape 约束 → 任何尺寸特征图都选它）
- input：`sme_fp32_mla_6x6`（6×6 tile，RequiresSME，注册表第 1）
- weight：`arm_fp32_4x4_3x3`（变换到 6×6，匹配 output 的 input_tile）

`n_multis = (4+3-1)² = 36` 个 GEMM，M = `ceil(OH/4)×ceil(OW/4)` tiles。SME 的 `MOPA`（矩阵外积累加）指令让变换与 GEMM 借助 2D tile 寄存器，吞吐远高于纯 SVE/NEON。

### 12.2 实际运行 case 逐条分析

下表对 11 个 case 做判定（IC = input C = weight 第 2 维；OC = weight 第 1 维；IH/IW = input H/W）。

| # | Input(N,C,H,W) | Weight(OC,IC,K) | Stride | IH | IW | IC | OC | 形状硬约束 | auto 回退触发项 | **auto 模式** | **显式 winograd** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (4,192,40,40) | (192,192,3,3) | 1,1 | 40 | 40 | 192 | 192 | ✓(s1,p1) | threads>28 | 回退→BRGEMM | ✅ acl:wino(SME, F4×4) |
| 2 | (4,768,80,80) | (768,768,3,3) | 2,2 | 80 | 80 | 768 | 768 | ✗ **stride≠1** | — | ✗→BRGEMM | ✗→BRGEMM |
| 3 | (4,96,80,80) | (96,96,3,3) | 1,1 | 80 | 80 | 96 | 96 | ✓ | threads>28 | 回退→BRGEMM | ✅ acl:wino(SME) |
| 4 | (4,768,40,40) | (768,768,3,3) | 2,2 | 40 | 40 | 768 | 768 | ✗ stride≠1 | — | ✗→BRGEMM | ✗→BRGEMM |
| 5 | (4,384,160,160) | (384,384,3,3) | 2,2 | 160 | 160 | 384 | 384 | ✗ stride≠1 | — | ✗→BRGEMM | ✗→BRGEMM |
| 6 | (4,48,160,160) | (48,48,3,3) | 1,1 | 160 | 160 | 48 | 48 | ✓ | **ih>112 && iw>112** (+threads) | 回退→BRGEMM | ✅ acl:wino(SME)* |
| 7 | (4,96,320,320) | (192,96,3,3) | 2,2 | 320 | 320 | 96 | 192 | ✗ stride≠1 | — | ✗→BRGEMM | ✗→BRGEMM |
| 8 | (4,192,20,20) | (192,192,3,3) | 1,1 | 20 | 20 | 192 | 192 | ✓ | threads>28 | 回退→BRGEMM | ✅ acl:wino(SME) |
| 9 | (4,384,80,80) | (384,384,3,3) | 2,2 | 80 | 80 | 384 | 384 | ✗ stride≠1 | — | ✗→BRGEMM | ✗→BRGEMM |
| 10 | (4,384,80,80) | (96,384,3,3) | 1,1 | 80 | 80 | 384 | 96 | ✓ | threads>28 | 回退→BRGEMM | ✅ acl:wino(SME) |
| 11 | (4,768,40,40) | (96,768,3,3) | 1,1 | 40 | 40 | 768 | 96 | ✓ | threads>28 | 回退→BRGEMM | ✅ acl:wino(SME) |

\* case 6 显式 winograd 时虽 `ih=iw=160>112`，但 auto 回退条件被 `&& convolution_auto` 门控跳过；形状硬约束（s1/p1/无dil/2D/groups=1）全满足，故 acl:wino 仍可生效（见 12.0 注）。

**统计**：
- stride=2 的 case（**2,4,5,7,9** 共 5 个）：acl:wino 在任何模式下都不可用（stride≠1 硬约束），一律走 `brgemm_conv<sve_512>`。
- stride=1 的 case（**1,3,6,8,10,11** 共 6 个）：auto 模式全因 `threads>28`（case 6 还因大图）回退到 BRGEMM；**显式 winograd 模式下这 6 个都能进 acl:wino**，用 SME F(4,4,3,3)。

### 12.3 显式 winograd 模式下 6 个 stride=1 case 的 GEMM 规模

F(4,4,3,3)：n_multis=36，输出 tile 4×4，stride=1 且 pad=1 时 OH=IH、OW=IW。

| # | OH=OW | n_tiles=ceil(OH/4)² | M | N(OC) | K(IC) | batch | 36 GEMM 总规模 (M×N×K×batch) |
|---|---|---|---|---|---|---|---|
| 1 | 40 | 10²=100 | 100 | 192 | 192 | 4 | 100×192×192×4 |
| 3 | 80 | 20²=400 | 400 | 96 | 96 | 4 | 400×96×96×4 |
| 6 | 160 | 40²=1600 | 1600 | 48 | 48 | 4 | 1600×48×48×4 |
| 8 | 20 | 5²=25 | 25 | 192 | 192 | 4 | 25×192×192×4 |
| 10 | 80 | 20²=400 | 400 | 96 | 384 | 4 | 400×96×384×4 |
| 11 | 40 | 10²=100 | 100 | 96 | 768 | 4 | 100×96×768×4 |

> case 6（160×160）M 最大（1600 tiles）但 N=K=48 最小，算术强度低，Winograd 变换加法占比高——这正是 auto 模式额外对 `ih>112` 回退的依据。case 11 的 K=768 最大，GEMM 占比高，Winograd 收益最显著。

### 12.4 端到端流程（显式 winograd，case 1：4×192×40×40）

```
1. prepare(一次性):
   weights[192,192,3,3] →permute HWIO→ →arm_fp32_4x4_3x3 权重变换→ V(6×6 Winograd域, 36 矩阵)
2. run(每次, N=4):
   src[NHWC, 4×192×40×40] →sme_fp32_mla_6x6 输入变换(SME)→ U(6×6域, 36 矩阵, M=100 tiles ×batch4)
   → 36 个 GEMM(100×192×192, SME/SVE) → M(Winograd域输出)
   → sme_fp32_mopa_4x4_3x3 输出变换(SME, +bias+ReLU) → dst[NHWC, 4×192×40×40]
3. post_ops.execute(): (若有 sum/binary)
```

### 12.5 NUMA 与线程模型要点

- **608 核跨 16 NUMA**：oneDNN 的 `dnnl_get_max_threads()` 通常返回总线程数（≈608），远超 28，故 auto 模式 acl:wino 全部回退。即便单 NUMA 绑定（38 核）仍 >28。
- **mutex 串行 vs 内部并行**：oneDNN 侧 `execute_forward` 全程持 `std::mutex`（`acl_winograd_convolution.cpp:29`），同一 primitive 实例并发多线程只能串行进入；但进入后 ACL 内部由 `NEScheduler` 用传入的 `nthreads` 多线程跑变换/GEMM。即"外层串行、内层并行"。
- **SME 是每核独立**：每个核有自己的 SME tile 寄存器，SME 变换/GEMM 可跨核并行，无锁争用（争用在 oneDNN resource_mapper 层）。
- **要真正启用 acl:wino**：设 `alg_kind=convolution_winograd`（推荐，绕过 auto 回退），或把 `OMP_NUM_THREADS` 压到 ≤28（牺牲并行度，不划算）。

### 12.6 何时不会命中 acl:wino（回退/拒绝场景汇总）

| 场景 | 拦截层 | 原因 | 落到 |
|------|--------|------|------|
| 本机 auto 模式（默认 608 线程） | auto 回退 | `threads=608>28` | `brgemm_conv<sve_512>` |
| 本机 auto 模式（nthr=16，实测） | auto 不回退 | `threads=16≤28` | **40 wino:acl + 19 brgconv**（见 `docs/onednn_comparison.md` §五） |
| stride=2 的 case（2,4,5,7,9） | 形状硬约束 | `stride≠1` | `brgemm_conv<sve_512>`（任何模式都不可用 wino） |
| auto + 大图（case 6 即便线程≤28） | auto 回退 | `ih=iw=160>112` | `brgemm_conv<sve_512>` |
| 带 groups | init_conf | 拒绝 groups | `brgemm_conv` 或 depthwise |
| threadpool 运行时 | init() 守卫 | `threading_runtime==threadpool` | `brgemm_conv` |

> **关键洞察**：在这台 608 核 + SME 机器上，**默认 auto 模式下 acl:wino 形同虚设**（线程数远超阈值），所有 case 实际命中 `brgemm_conv<sve_512>`。但**用 `DNNL_NUM_THREADS=16` 限制线程数后（nthr=16 ≤ 28），auto 不回退**——实测 59 shape 中 40 个选 wino:acl（见 `docs/onednn_comparison.md` §五）。且我们的 F(4,4) Winograd 全面碾压 oneDNN 的 wino:acl 1.15-9.5x。acl:wino 的价值在于：当用户对 stride=1 的 3×3 层显式指定 `winograd` 算法时，借助 SME 的 F(4,4,3,3) 获得乘法量下降的红利。

---

## 13. 性能特性与局限

### 13.1 优势

- **乘法量降低**：F(2,2,3,3) 把每输出像素等效乘法从 9 降到 4（≈44%）；F(4,4,3,3) 进一步降（36 GEMM 覆盖 16 像素，等效 36/16=2.25 次/像素）。
- **权重复用**：权重变换在 prepare 一次性完成，run 时直接用变换后权重，核窗口累加代价已付清。
- **batched GEMM 高吞吐**：变换后用 arm_gemm 的 SVE 内核，复用成熟 GEMM 优化。
- **融合**：bias + ReLU 融在输出变换，省一次访存。

### 13.2 劣势与局限

- **形状受限**：仅 stride=1、pad≤1、无 dilation、2D、无 groups、小核（3×3/5×5/1×N）。
- **mutex 串行**：整个 run 全局锁，高并发多线程场景受限（auto 模式线程>28 回退）。
- **额外内存**：Winograd 域 U/M/V 矩阵 + per-thread 工作空间，比直接卷积多一块 scratchpad（`TransformedInput/Output/WorkspaceIO`）。
- **精度**：fast_math=true，F(4,4,5×5) 等配置可能精度略降；fp16 在大输入值时中间结果可能溢出（`NEWinogradConvolutionLayer.h:94-95` 警告）。
- **加法开销**：变换的加法量随 tile 变大而上升，小特征图时可能抵消乘法节省（这正是 `ih/iw>112 回退` 的依据）。
- **只前向**：ACL Winograd 不支持反向传播。

### 13.3 与 BRGEMM 的取舍

| 维度 | acl:wino | brgemm_conv<sve_512> |
|------|----------|----------------------|
| 算法 | Winograd 变换减乘法 | im2col+GEMM 融合（不物化 col） |
| 形状 | 窄（3×3s1 小图） | 宽（任意形状 NHWC） |
| 多线程 | mutex 串行 + 内部多线程 | 无锁，按维度切分并发 |
| 内存 | 额外 Winograd 域矩阵 | BRGEMM batch（无 col 物化） |
| 融合 | bias+ReLU（限 RELU 族） | bias+sum+eltwise+binary+量化（更全） |
| 适用 | 小核小图、线程少、显式 wino | 通用主力 |

分发表把 acl:wino 排第 1、brgemm 排第 4：**优先尝试 wino 的乘法优势，但 auto 模式用回退条件把"winograd 不划算"的形状（大图/小通道/多线程）让给 BRGEMM**。

---

## 14. 关键文件索引

### oneDNN 侧（`src/cpu/aarch64/`）

| 文件 | 关键内容 |
|------|----------|
| `acl_winograd_convolution.hpp:68-145` | `acl_wino_convolution_fwd_t` primitive；`pd_t::init()`（:75）；`configure()`（:44）；`execute_forward` 调用（:136） |
| `acl_winograd_convolution.cpp:25-39` | `execute_forward`：mutex + import_memory + run + post_ops |
| `acl_convolution_utils.hpp:41-62` | `acl_conv_conf_t` 结构；`acl_obj_t` 模板 |
| `acl_convolution_utils.hpp:82-108` | `init_scratchpad`：ACL workspace → oneDNN scratchpad |
| `acl_convolution_utils.hpp:187-232` | `execute_forward_conv_acl`：import_memory + ITensorPack + run |
| `acl_convolution_utils.cpp:36-285` | `acl_init_conf`：desc→TensorInfo/PadStrideInfo/dilation 翻译；wino 提前 return（:206） |
| `acl_convolution_utils.cpp:287-334` | `init_conf_wino`：auto 回退条件 + 形状约束 + validate |

### ACL 侧（`D:\300Code\ComputeLibrary-53.1.0\`）

| 文件 | 关键内容 |
|------|----------|
| `arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h:52-128` | 运行时封装；configure/validate/run/prepare；支持的 kernel 尺寸（:84） |
| `src/cpu/operators/CpuWinogradConv2d.h:43-140` | `CpuWinogradConv2d` 类；AuxTensorIdx（:104）；成员（:116-137） |
| `src/cpu/operators/CpuWinogradConv2d.cpp:174-322` | `configure()`：选变换（:192）、算 GEMM 维度（:223）、构造 Winograd 域 TensorInfo（:230）、aux_mem（:306） |
| `src/cpu/operators/CpuWinogradConv2d.cpp:323-357` | `validate()`：fp16 需 fast_math（:336） |
| `src/cpu/operators/CpuWinogradConv2d.cpp:359-424` | `run()`：permute→input_transform→GEMM→output_transform→permute→activation |
| `src/cpu/operators/CpuWinogradConv2d.cpp:426-475` | `prepare()`：permute weights + weight_transform + gemm prepare（一次性） |
| `src/core/NEON/kernels/assembly/winograd.hpp:240-270` | `WinogradImpl` 结构（三元组 + gemm_args + winograd_spec）；`get_implementation` 声明 |
| `src/core/NEON/kernels/convolution/winograd/winograd_implementations.hpp:236-339` | `get_implementation`：贪心匹配算法 + GEMM 参数计算（:299-320）+ WinogradDomainSpec（:322-336） |
| `.../winograd/input_transforms_fp32.cpp:51-67` | 输入变换注册表（sme/sve/a64 6×6, 4×4, 1×8） |
| `.../winograd/output_transforms_fp32.cpp:50-66` | 输出变换注册表（4×4_3×3[+SME/LargerShape], 2×2_3×3, 2×2_5×5, 1D） |
| `.../winograd/weight_transforms_fp32.cpp:49-64` | 权重变换注册表（4×4_3×3, 2×2_3×3, 2×2_5×5, 1D） |
| `.../winograd/input_transform.hpp:54-377` | `TransformBase`/`TransformUnpadded` 驱动；tile 遍历（:93）；padding 处理（:303） |
| `.../winograd/weight_transform.hpp:37-141` | 权重变换驱动；多线程按 16 通道分组（:60） |
| `.../winograd/weight_transforms/arm_fp32_2x2_3x3.cpp:60-83` | **G 矩阵源码**（F(2,2,3,3)） |
| `.../winograd/input_transforms/arm_fp32_4x4.cpp:99-128` | **B^T 矩阵源码**（F(2,2,3,3) 输入变换） |
| `.../winograd/output_transforms/arm_fp32_2x2_3x3.cpp:64-104` | **A^T 矩阵源码** + fused bias+ReLU（F(2,2,3,3) 输出变换） |

---

## 附：F(2,2,3,3) 三矩阵速查卡

```
权重变换  V = G · g · G^T  (3×3核 → 4×4域)
        ┌ 1     0     0   ┐
   G =  │ 0.5   0.5   0.5 │
        │ 0.5  -0.5   0.5 │
        └ 0     0     1   ┘

输入变换  U = B^T · d · B   (4×4 tile → 4×4域)
          ┌ 1  0 -1  0 ┐
   B^T =  │ 0  1   1  0 │
          │ 0 -1   1  0 │
          └ 0  1   0 -1 ┘

输出变换  f = A^T · M · A   (4×4域 → 2×2输出, +bias +ReLU)
          ┌ 1  1  1  0 ┐
   A^T =  │           │
          └ 0 -1 -1 -1 ┘

GEMM:   n_multis = (2+3-1)² = 16 个, M=tiles, N=OC, K=IC
        每输出像素等效乘法 9→4 (≈44%)
```

> 本文基于 oneDNN 3.12.1 + ACL 53.1.0 源码梳理，矩阵系数与函数签名均逐行核对，行号供定位参考，具体实现以源码为准。
