# oneDNN 对照（e2e + benchdnn）：数据、工具、判读

> 本仓库（arm_gemm 53.1.0 后端，SVE-512）与 oneDNN 3.12.1（鲲鹏 920F 原生 build）的
> 性能对照。本文是**数据入口**：数据怎么生成、放哪、怎么判读、坑在哪。给后续优化
> agent 与人工判读的索引。
>
> 对照基准用例：`shapes/conv_all.csv`（59 行，全部 stride=1/group=1/3x3/pad=1）。

## 一、为什么对照 / 口径

- 我们 = Winograd F(4,4,3,3) + arm_gemm（`build/bench_winograd`）；oneDNN = 独立实现
  （AArch64 下走 ACL `wino:acl` 或 `brgconv`）。
- **同 shape、同绑核（16 线程，`OMP_PROC_BIND=close OMP_PLACES=cores`）、primitive
  跨迭代复用**（e2e 侧）。加速比读法：**比值 >1 = 我们快**。
- ⚠️ **benchdnn 与端到端测速不是一回事**（同实现可差 1.7~3.6x，见
  `docs/why_faster_than_acl_23.11.md` §10.4 与 `shapes/README.md`）。所以对照分两列：
  e2e 列（与 ours 同口径）和 benchdnn 列（oneDNN 自带 driver，口径不同、仅参考）。

## 二、工具链（`tools/onednn/`）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `run_onednn_e2e.sh` | 编译 `onednn_e2e.cpp` 并对 CSV 计时（primitive 复用，算法梯子 direct→auto→winograd） | `build/onednn_e2e.csv`（`mb,ic,ih,iw,oc,onednn_ms`） |
| `run_benchdnn.sh` | 对 `shapes/conv_all.list` 跑 `benchdnn --conv --reset [--alg=WINO]` | `build/benchdnn_{auto,wino}.txt` |
| `merge_onednn.sh` | 合并三列 → `mb,ic,ih,iw,oc,ours_ms,onednn_e2e_ms,benchdnn_wino_ms[,benchdnn_auto_ms]` + 加速比 | stdout（A6 打印） |
| `ab_onednn.sh` | **单 sbatch 作业**跑全套：P0 状态探针 → A1 ours → A2 e2e → A3/A4 benchdnn → A5 filter_sweep → A6 merge → P1 | 作业 stdout |

关键约定：
- **所有对比必须塞进同一个 sbatch 作业**：node03 跨作业存在 3~7x 性能态，跨作业数字
  一律作废。统一入口：`sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"`。
- 可选环境变量：`ONEDNN_ROOT`（oneDNN 头文件+lib 目录）、`BENCHDNN`（benchdnn 路径）、
  `SKIP_AUTO=1`（跳过 A4）、`SKIP_FILTER=1`、`T=<线程数>`（默认 16）。
- 两个 oneDNN 实体**不是同一构建**：e2e 链接 `/workspace/z00889957/000Libs/onednn-3.12.1-release`
  的 `libdnnl.so`；benchdnn 是 `/workspace/z00889957/000Libs/oneDNN-3.12.1/build` 的
  可执行。ab_onednn 会打印两边 `ldd` 的 libdnnl 链接，判异前先看它。
- 集群代码是 scp 文件副本（无 `.git`），CRLF 由脚本 `sed -i 's/\r$//'` 自愈。

## 三、怎么跑（集群，完整一轮）

```bash
sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"
# 作业输出里 A6 直接打印合并表；把整份输出贴回会话即可判读。
```

只重放合并（产物已在 build/）：`bash tools/onednn/merge_onednn.sh build/ours_cmp.csv
build/onednn_e2e.csv build/benchdnn_wino.txt build/benchdnn_auto.txt shapes/conv_all.csv`

## 四、⚠️ e2e 系统性 OOM 血泪史（新 agent 必读，勿重蹈）

> 根因与判读见下；**弯路本身、为什么第一理论错了、可复用方法论**见
> `docs/debugging_lessons.md`。

- **现象**（2026-08-27~29）：e2e 里每个 conv 的 PD 创建都 `out_of_memory`，59 个形状、
  any/nchw 全挂；[preOMP] ok、[smoke] FAIL。
- **错修**（8244944）：误判为 `omp_set_num_threads` 触发，删掉该调用。集群实测
  **证伪**：nothr=608 线程、无 OMP_NUM_THREADS env、`ONEDNN_MAX_CPU_ISA=advanced_simd`
  （SVE off）三种配置下 [smoke] 依旧全 oom —— 与线程数/ISA 无关。
- **真根因**：`onednn_e2e.cpp` 给每个**无膨胀** conv 显式传了 `dilates={1,1}`
  （CSV 的 dh=dw=0，应为 {0,0}）。集群 `onednn-3.12.1-release` 的 `conv_desc_init`
  **不做 dst 维度校验**，错误 dilation 直达 impl → 巨大分配 → 系统性 PD 创建 OOM。
  （本地下载的 3.12.1 源码 `convolution.cpp:166` 是 `VCHECK_CONV → invalid_arguments`，
  与集群实测 oom **不符** ⇒ 集群 lib ≠ 本地源码，**一切以集群实测为准**。）
- **红鲱鱼**：`eltwise + format_any` 在该 build 本身不可用（本地源码
  `eltwise.cpp:65-66` 直接 reject）——[preOMP]/[smoke] 的差异是**格式**（nchw vs any），
  不是「第 1 个 PD 好、后面中毒」的顺序问题。
- **修复**（4bdc3ee，按用户参考程序格式）：conv PD 一律走「带 bias、不带 dilates」
  的公开重载（dnnl.hpp 5740+ 第一个公开 ctor，内部 dilates 置 0）；bias 恒用具体格式
  `tag::x`（不用 format_any）；算法梯子默认 direct 放首位（= 参考程序 proven 组合
  direct+fwdinf 先试）；探针移到数据流之后。

### 诊断矩阵（run_onednn_e2e.sh 每次都会打印，据此判读）

| 探针 | 预期 | 含义 |
|---|---|---|
| `conv_direct_any_nodil` | ok | 修复后的生产配置（any 布局、无 dilates）可建 PD |
| `conv_direct_any_dil1` | FAIL | 旧 dilates={1,1} 复现 —— 两者对照即证实根因 |
| `eltwise_any` | FAIL | 预期不可用（该 build 不支持 eltwise+format_any），非数据路径 |
| `[smoke]` | ok | eltwise-nchw 的**第 2 个** PD 不中毒（验证无跨 PD 状态污染） |
| `[env]` | 3.12.1 | 版本一致性 |
| `[thr]` | = env 线程数 | 确认 OMP_NUM_THREADS 生效 |
| `[capi]/[heap]` | ok | C API / malloc 侧探针 |

判读结论：**`conv_direct_any_nodil=ok` 且 `conv_direct_any_dil1=FAIL` ⇒ 根因确认为
dilates 旧 bug，修复生效**，e2e 应出全量 59 行数据。

## 五、数据与结论（最终 2026-08-31）

### ours vs OpenBLAS —— 已终局 ✅

59/59 全胜，geomean **1.99x**，合计 321.9 vs 804.9 ms = **2.50x**；大形状折算 GEMM
≈960 GFLOPS/16 核 ≈ SVE-512 峰值 94%。**全量逐形状表见 `docs/final_benchmark_bfd6b1e.md`。**

### ours vs oneDNN —— 已闭环 ✅

三列同作业、同 16 线程（`numactl -C 0-15` + `DNNL_NUM_THREADS=16`）、同 node03：

| 数据 | 状态 | 位置 |
|---|---|---|
| A1 ours 59 形状 | ✅ | `build/ours_cmp.csv` |
| A2 e2e 59 行 | ✅ ACL 库路径修复 + CSV 清空 | `build/onednn_e2e.csv` |
| A3 benchdnn WINO | ✅ `ONEDNN_VERBOSE=1` + `--mode=p` + `-v4` + exec 行解析 | `build/benchdnn_wino.txt` |
| A4 benchdnn auto | ✅ 同上 | `build/benchdnn_auto.txt` |
| A5 filter_sweep | ✅ 已闭环 | `build/filter_sweep_{auto,6x4VL,8x1VL,inter}.csv` |
| A6 合并表 | ✅ `src=59 na=0` 三列全有数据 | `build/merged.csv` |

### 最终对照结果

三列含义不同，回答不同问题：

| 对照列 | oneDNN 算法 | 问题 | 我们快的情况 | oneDNN 快的情况 |
|--------|-------------|------|-------------|-----------------|
| **e2e**（主对照） | brgconv:sve_512 | 我们比 oneDNN 最优自动选择快还是慢？ | 大形状 1.1-2.8x | 小形状 0.09-0.91x |
| **benchdnn_wino** | wino:acl | 同样做 Winograd，我们比 ACL 快还是慢？ | **全部 1.2-9x** | 无 |
| **benchdnn_auto** | brgconv:sve_512 | benchdnn driver 口径的 brgconv 对照 | 大形状 1.1-3.1x | 小形状 0.10-0.85x |

**关键结论**：
1. **e2e 列**（同口径主对照）：分水岭在 IC×spatial ≳ 256×56²。大形状我们的 F(4,4) Winograd
   摊薄变换开销后反超 brgconv；小形状 brgconv 无变换开销直接更快。
2. **benchdnn_wino 列**（同算法 PK）：我们的 F(4,4) 全面碾压 oneDNN 的 wino:acl（疑似
   F(2,2)），1.2-9x。原因：F(4,4) 比 F(2,2) 少 ~44% GEMM 调用 + 单次 GEMM 矩阵更大
   cache 友好 + arm_gemm JIT SVE 内核优于 oneDNN 的 gemm_api。
3. **benchdnn_auto ≈ e2e + ~10%**：两列都是 brgconv:sve_512，差异来自不同 oneDNN build
   + 不同 measurement driver（benchdnn `--mode=p` best-of-N vs e2e best-of-20）。
4. benchdnn 自检确认 `nthr=16`（`DNNL_NUM_THREADS` + `OMP_THREAD_LIMIT` 生效，不再 608 超订）。

### filter_sweep（M=25 选核）—— 已闭环 ✅

结论：**auto 保持最优，不改选核**。20² 族 auto 距各行最优 ≤5%；大形状 4,384,160²,384
强选 8x1VL 慢 1.45x、inter 慢 1.23x、6x4VL 最优且 auto 已选。唯一可挖点（不足以全局
改）：4,96,80²,96 上 8x1VL 快 ~19%、4,768,20²,96 快 ~5%。

## 六、benchdnn [NO-SRC] 已闭环——修复历程与弯路

> **弯路本身的详细复盘见 `docs/debugging_lessons.md` 第二节。** 本节只记最终方案。

### 最终方案（run_benchdnn.sh 关键配置）

```bash
# 1. KNOWN_GOOD BIN 路径优先（避免 find 选错二进制）
# 2. : > "$OUT" 前移清空（避免 set -e 失败时残留旧文件）
# 3. ONEDNN_VERBOSE=1 + --mode=p + -v4（拿卷积纯执行时间，多次迭代取 MIN）
# 4. DNNL_NUM_THREADS=16 + OMP_THREAD_LIMIT=16（限制 oneDNN 线程数，不用 608）
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS \
    DNNL_NUM_THREADS=$THREADS OMP_THREAD_LIMIT=$THREADS \
    ONEDNN_VERBOSE=1 \
    "${RUNNER[@]}" "$BIN" --conv --mode=p -v4 --reset "${ALG[@]}" \
    --batch=shapes/conv_all.list >"$OUT" 2>&1
```

merge 三级兜底：**perf 行 → exec 行 → PASSED 行**。集群实测 perf_lines=59 +
exec_lines=57131 + nthr=16，merge `src=59 na=0`。

### 弯路摘要（详见 `docs/debugging_lessons.md` 第二节）

| # | 弯路 | 耗时 | 真根因 |
|---|------|------|--------|
| 1 | 误判 CRLF 导致脚本无输出 | 2 轮 | 实为 `shift` 多余导致 `set -e` 静默退出 |
| 2 | `--mode=p` 不出 perf 行 → 加 `ONEDNN_VERBOSE=exec` | 2 轮 | env var 值应为 `=1` 不是 `=exec`；且需 `-v4` flag 配合 |
| 3 | `-v4` 在集群 3.12.1 上无输出 | 1 轮 | 需 `export ONEDNN_VERBOSE=1` 才启用 verbose（3.4.0 自动启用） |
| 4 | e2e 0 行 + ldd 无 dnnl | 2 轮 | `libarm_compute.so` 不在 `LD_LIBRARY_PATH`（oneDNN 链接 ACL） |
| 5 | e2e 输出混入 verbose 行 | 1 轮 | `ONEDNN_VERBOSE=1` 污染 e2e stdout，加 `ONEDNN_VERBOSE=0` 给主运行 |
| 6 | `grep -c \|\| echo 0` 产生 `0\n0` | 1 轮 | grep -c 无匹配时打印 0 且返回 1，echo 再打印 0 |

## 七、给优化 agent 的注意事项

（注意事项已移至 §八末尾，此处仅保留索引。完整作业日志可能过长——判读只取
`build/SUMMARY.txt`，交接 SOP 见 §八。）

## 八、交接 SOP（完整日志太长时只取判读文件）

判读/交接**只需 `build/SUMMARY.txt` 一份**，包含：
数据行数（ours/e2e/benchdnn）、e2e impl 直方图、诊断矩阵（build/diag.txt）、合并表
（build/merged.csv）。需要更细时再补：

| 文件 | 内容 |
|---|---|
| `build/SUMMARY.txt` | ★ 判读入口（ab_onednn.sh 末尾自动生成） |
| `build/merged.csv` | 合并表（ours/e2e/benchdnn + 加速比） |
| `build/diag.txt` | 单 PD 诊断矩阵 |
| `build/onednn_e2e.csv` / `.err` | e2e 全量 59 行 / impl 直方图 + 探针 |

**别的机器上的 agent 重跑同一套**：
1. `scp` 整树（含 `tools/onednn/`、`shapes/`，无 .git，CRLF 脚本自愈）到集群；
2. `sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"`（或
   `SKIP_AUTO=1 SKIP_FILTER=1 bash ...` 快速档，仅跳 A4/A5）；
3. 读 `build/SUMMARY.txt`，按下列清单判读：
   - **诊断矩阵**：`conv_direct_any_nodil=ok` 且 `conv_direct_any_dil1=FAIL` ⇒ OOM 根因
     确认修复（`eltwise_any=FAIL` 属预期，该 build 不支持 eltwise+format_any）。
   - **e2e 数据行数 = 59** ⇒ 修复生效，e2e 不再 oom。
   - **impl 直方图**：`jit:sve`/`brgconv:sve_512` 为主 ⇒ oneDNN 走了 SVE 路径；出现
     `jit:asimd` ⇒ SVE 未生效（检查 ONEDNN_MAX_CPU_ISA / 编译）。
    - **合并表加速比**：`onednn/ours`、`benchdnn_*/ours` >1 即我们快。benchdnn 列
      是 `ONEDNN_VERBOSE=1` + `--mode=p -v4` 拿的卷积纯执行时间（exec 行解析，
      perf→exec→PASSED 三级兜底）。merge_summary 应显示 `src=59 na=0`。
      自检行 `nthr=16`（不是 608）确认线程数正确。
    - **P0/P1 频探**：主频应稳定 2.0GHz（606/608 核），漂移说明节点状态异常。

  1. **同作业才可比**：任何跨作业数字（含本文历史数字）只作定性参考，性能结论必须
     `sbatch -w node03 --exclusive` 单作业内 A/B。
  2. **ours 侧无 debug 才可信**：`WINO_GEMM_DEBUG=1`（E1）会不成比例放大小形状计时，
     新旧/大小对比只信 E3/E5 无 debug 读数。
  3. **修正旧文档口径**：README/AGENTS 里「8/9 case 超越 oneDNN」「6/9 超越」是
     **OpenBLAS 后端 + benchdnn WINO 口径**的历史数字（oneDNN 被测量方式拖慢 1.7~3.6x），
     不能当 arm_gemm 后端的结论；arm_gemm 后端 vs OpenBLAS 才是 59/59 终局。
  4. **oneDNN WINO 依赖 ACL**：benchdnn 链接 `/data1/.../ComputeLibrary-53.1.0`（带 ACL），
     e2e 链接 `/workspace/.../ComputeLibrary-53.1.0`（也带 ACL）。两个 build 不同但都有 ACL，
     wino:acl 列可用。e2e 用 `--auto` 在 16 线程下全选 brgconv:sve_512（不走 wino）。
  5. **git 边界**：不要 stage 用户未提交的 `README.md` 与 `swish_sve/`；
     `tools/diag_verbose.sh` 是临时诊断脚本，闭环后已删除。
