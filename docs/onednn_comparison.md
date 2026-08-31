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

## 五、数据与结论（现状 2026-08-29）

### ours vs OpenBLAS —— 已终局 ✅

59/59 全胜，geomean **1.99x**，合计 321.9 vs 804.9 ms = **2.50x**；大形状折算 GEMM
≈960 GFLOPS/16 核 ≈ SVE-512 峰值 94%。**全量逐形状表见 `docs/final_benchmark_bfd6b1e.md`。**

### ours vs oneDNN —— 进行中（⚠️ benchdnn 列卡在 [NO-SRC]，见 §六）

| 数据 | 状态 | 位置 |
|---|---|---|
| A1 ours 59 形状 | ✅ 有 | `build/ours_cmp.csv` |
| A2 e2e 59 行 | ⚠️ 08-31 作业 csv **1423 行**（应 59，疑过期，merge last-wins 值仍可用） | `build/onednn_e2e.csv` |
| A3 benchdnn WINO | ⚠️ 文件 59 行但**无 perf 行**（[NO-SRC]，待解，见 §六） | `build/benchdnn_wino.txt` |
| A4 benchdnn auto | ⚠️ 同上 | `build/benchdnn_auto.txt` |
| A5 filter_sweep | ✅ 已闭环 | `build/filter_sweep_{auto,6x4VL,8x1VL,inter}.csv` |
| A6 合并表 | ⚠️ benchdnn 两列 N/A；e2e 列可用 | `build/merged.csv` |

已观察到的三点（**均待本轮重跑核实**）：
1. **benchdnn 列此前虚高已三修（50f9f2a 终版）**：
   - ① 线程：sbatch --exclusive 下默认 608 线程超订（小形状慢 100~900x）。但
     `OMP_NUM_THREADS=16` 对带 TBB 的 ACL build **无效**（重跑数值与修复前逐位
     相同）——真正有效的是 **`numactl -C 0-15` 绑核**（用户实测）。脚本已改
     numactl 优先（`BENCHDNN_NUMACTL` 可覆盖，如 `"-C 0-15 -m 16"`）。
   - ② 计时模式：benchdnn 默认 corr 模式，`PASSED (N ms)` 是含 fill/ref/compare
     的**聚合**时间，不可比。必须 `--mode=p` 才打印
     `perf,<engine>,<impl>,<name>,<prb>,...,<min-ms>,...` 行——`%-time%`
     = 逐次 start/stamp 的最小**单次执行** ms（已读 oneDNN 源码
     tests/benchdnn/utils/perf_report.hpp + measure_perf_individual 确认）。
   - ③ merge：只认 perf 行（优先）与 ONEDNN_VERBOSE exec 行（兜底）；两者都无 →
     整列 N/A（彻底不用 PASSED 聚合时间），merge_summary 标 `[NO-SRC]`。
     ⚠️ benchdnn 的 perf `%prb%` 字段是**缩写描述符**（相邻相等对省略后者：
     ih==iw 时打印 `mb4ic192ih40oc192oh40kh3ph1...`，无 `iw`/`ow`/`kw`）。
     merge 须对 `iw` 缺失回退 `iw=ih`（已修，本地合成 perf 行 + 集群实测
     batch perf 输出验证）。配套 bug：BD_B 嗅探原用 `r[0-9]+"` 匹配 PASSED
     行，但 `--mode=p` 不打印 PASSED 行，已改为 `^perf,|r[0-9]+"`。
     （注：2026-08-29 首跑 [NO-SRC] 的真因是**该作业跑了旧脚本**——build 文件
     是 corr 模式 `0:PASSED (613 ms)`，根本没 perf 行；手动 `--mode=p` batch
     实测 perf 行正常。）
2. **e2e 列 = 最终对照**（同库 onednn-3.12.1-release、同 16 线程绑核、单次执行）：
   初步趋势——oneDNN brgconv:sve_512（GEMM 卷积）小形状更快（0.35~0.87x），
   我们 F(4,4,3,3) 在大而深形状反超（4,384,80²,96 1.49x / 4,768,40²,96 1.78x）。
   注意 benchdnn（wino:acl，另一份带 ACL 的 lib）与 e2e（brgconv:sve_512，无 ACL）
   **不是同一个 build**，benchdnn 列只作 Winograd 算法族参考。
3. A2 e2e 此前全 oom，4bdc3ee 修复后已出 59 行（本表），本次作业核心产出。

### filter_sweep（M=25 选核）—— 已闭环 ✅

结论：**auto 保持最优，不改选核**。20² 族 auto 距各行最优 ≤5%；大形状 4,384,160²,384
强选 8x1VL 慢 1.45x、inter 慢 1.23x、6x4VL 最优且 auto 已选。唯一可挖点（不足以全局
改）：4,96,80²,96 上 8x1VL 快 ~19%、4,768,20²,96 快 ~5%。

## 六、⚠️ 当前卡点（2026-08-31）：benchdnn 仍 [NO-SRC]，待解

### 已确认事实
- 集群已部署新脚本：`run_benchdnn.sh` 含 `--mode=p`（grep=5），`merge_onednn.sh` 含
  `iw=ih` 回退（53/74 行）——**部署无误**。
- 手动**单形状**与手动**batch**均能出 `perf,` 行（node02 实测）：
  `numactl -C 0-15 $BENCHDNN --conv --mode=p --reset --alg=WINO --batch=shapes/conv_all.list`
- 但 sbatch 作业（node03，08-31 09:04）产出的 `build/benchdnn_{wino,auto}.txt` 仍**无 perf 行**
  （`grep -c '^perf,'` = 0），merge `src=0 na=0 [NO-SRC]`。
- 08-29 首跑 [NO-SRC] 的真因已实锤 = **旧脚本**（corr 模式 `0:PASSED (613 ms)`，无 --mode=p）。
- 08-31 作业 e2e 列异常：`onednn_e2e.csv` **1423 行**（应 59）。`parse_csv`
  （onednn_e2e.cpp:346 每 shape 恰 1 行）对 59 形状不可能出 1423 → 疑**过期/污染文件**
  （merge 用 last-wins，表内 e2e 值仍可用，但须核实）。
- 诊断矩阵正常：`conv_direct_any_nodil=ok` / `conv_direct_any_dil1=FAIL`（OOM 根因修复确认有效）。
- benchdnn 链接 `/data1/z00889957/apps/oneDNN-3.12.1/build/src/libdnnl.so.3`（第三处，
  与 e2e 的 `onednn-3.12.1-release` 不是同一构建）。

### 卡点假设（按嫌疑排序）
1. **作业内 BIN 探测选错二进制**：`run_benchdnn.sh` 用 `find ... | head -1` 探测
   （cand: ONEDNN_ROOT /usr/local /opt /workspace/z00889957/000Libs /workspace）。
   已知好路径 `/workspace/z00889957/000Libs/oneDNN-3.12.1/build/tests/benchdnn/benchdnn`
   （相对 000Libs 深度 5，`-maxdepth 5` 可命中）。若作业环境下别的 cand 先命中
   **更旧的 benchdnn**（不认 `--mode=p`）→ corr 模式 PASSED 行 → merge [NO-SRC]。
2. **读取过期文件**：ab_onednn `set +e`，若 run_benchdnn.sh 静默失败，
   build/benchdnn_wino.txt 残留上一轮 corr 文件被 merge 照读。
3. 作业上下文差异（env/绑核/sbatch）破坏 perf 输出（可能性最低——手动 batch 同参数可出）。

### 待解诊断（已向用户索取）
- `tail -8 build/benchdnn_wino.txt`：末尾自检行 `# PASSED=… perf_lines=…`（新脚本才写）。
- `grep -c '^perf,' build/benchdnn_*.txt` + 文件 mtime（应为本作业 09:04 Aug 31）。
- 作业日志 A3/A4 的 `[benchdnn] binary:` / `PASSED lines:` / `!! WARNING` 行
  （`ls -t slurm-*.out | head -1` 取最新日志）。

### 下一步（诊断后）
- BIN 选错 → `run_benchdnn.sh` 改**固定已知路径**优先 + 打印解析结果 + 自检行带 binary
  路径（job 日志不可靠，关键行写入 OUT 文件）。
- 过期文件 → 运行前 `: > "$OUT"` 强制清空。
- e2e 1423 行 → 核实 mtime / 确保 tee 前截断；若编译失败读了旧 CSV 需顺带修。
- 出 exec 行后重跑整作业：merge 应 `src≈59 na=0`，benchdnn 列与 e2e 列同量级。
- 终态判读：benchdnn(wino:acl, ACL 构建) / e2e(brgconv:sve_512) / ours 三列同作业。

### ✅ 已实施防御性修复（本地仓库，待集群验证）

三假设的防御性修复已落地（commit 待 push），同时覆盖 ① BIN 选错 + ② 过期文件 + perf 模式不可用：

| 改动 | 文件 | 作用 |
|------|------|------|
| KNOWN_GOOD 优先 | `run_benchdnn.sh:57-60` | `/workspace/z00889957/000Libs/oneDNN-3.12.1/build/tests/benchdnn/benchdnn` 先于 find 循环检查，避免 `/usr/local`、`/opt` 先命中旧 benchdnn |
| `: > "$OUT"` 前移 | `run_benchdnn.sh:50` | OUT 赋值后立即清空，gen_benchdnn_list / BIN 探测 / numactl 任一步 `set -e` 失败都不留旧文件（ab_onednn `set +e` 读到空文件 → merge [NO-SRC]，不读过期数据） |
| gen_benchdnn_list 非致命 | `run_benchdnn.sh:75` | `\|\| true` 防止刷新描述符失败时 `set -e` 提前退出 |
| `--mode=p` → `ONEDNN_VERBOSE=exec` | `run_benchdnn.sh:89-90` | 集群 benchdnn build 多次实测无 perf 行（可能不支持 perf 模式），改用 `ONEDNN_VERBOSE=exec` 让 oneDNN 库直接打印 per-execution 计时行 |
| 自检带 binary | `run_benchdnn.sh:107` | `# run_benchdnn self-check: binary=$BIN ...` 写入 OUT 末尾，SUMMARY 直接可见用了哪个 benchdnn |
| N_EXEC grep 放宽 | `run_benchdnn.sh:100` | 从 `,primitive,exec,` 改为 `(onednn\|dnnl)_verbose,.*exec`，与 merge 解析器一致 |
| merge exec 解析器修 | `merge_onednn.sh:67-85` | regex 放宽（认 `dnnl_verbose` 旧名 + 不要求 "primitive"）+ 描述符从整行 $0 搜（不限于单字段）+ 认 `mb:4` 冒号格式 |
| merge PASSED 兜底 | `merge_onednn.sh:85-110` | 三级兜底：perf → exec → PASSED。PASSED 行 `N:PASSED (XXX ms) __REPRO: ...` 的聚合时间（含 fill/ref/compare）作最后参考，比 [NO-SRC] 强 |
| e2e CSV 清空 | `run_onednn_e2e.sh:62` | `mkdir -p build` 后 `: > build/onednn_e2e.csv`，编译失败不留 1423 行旧文件 |

**集群验证（部署后跑一次 sbatch 作业，检查这 4 项）**：
1. `tail -3 build/benchdnn_wino.txt` —— 末尾自检行应有 `binary=/workspace/.../benchdnn` + `exec_lines`/`PASSED` 计数。
2. `grep -cE '(onednn|dnnl)_verbose,.*exec' build/benchdnn_*.txt` —— 若 >0 则 merge 用 exec 行（单次执行口径）；若 =0 则 merge 回退 PASSED 聚合时间。
3. `head -1 build/benchdnn_wino.txt` —— 应是 `[benchdnn] binary: /workspace/z00889957/000Libs/oneDNN-3.12.1/build/tests/benchdnn/benchdnn`。
4. `wc -l build/onednn_e2e.csv` —— 应 59 行（+1 行表头），不再是 1423。

## 七、给优化 agent 的注意事项

（见下方；另：完整作业日志可能过长——判读只取 `build/SUMMARY.txt`，交接 SOP 见 §八）

### 注意事项

1. **同作业才可比**：任何跨作业数字（含本文历史数字）只作定性参考，性能结论必须
   `sbatch -w node03 --exclusive` 单作业内 A/B。
2. **ours 侧无 debug 才可信**：`WINO_GEMM_DEBUG=1`（E1）会不成比例放大小形状计时，
   新旧/大小对比只信 E3/E5 无 debug 读数。
3. **修正旧文档口径**：README/AGENTS 里「8/9 case 超越 oneDNN」「6/9 超越」是
   **OpenBLAS 后端 + benchdnn WINO 口径**的历史数字（oneDNN 被测量方式拖慢 1.7~3.6x），
   不能当 arm_gemm 后端的结论；arm_gemm 后端 vs OpenBLAS 才是 59/59 终局。
4. **oneDNN WINO 依赖 ACL**：本原生 build（无 ACL）可能全 unimplemented → wino 列全
   N/A，以 auto 列为主。
5. **git 边界**：不要 stage 用户未提交的 `README.md` 与 `swish_sve/`；`tools/diag_ab.sh`
   是临时诊断脚本，闭环后删除。

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
   - **合并表加速比**：`onednn/ours`、`benchdnn_*/ours` >1 即我们快；benchdnn 列
     **应为 perf 单次行**（--mode=p + numactl 绑核，50f9f2a），与 e2e 列同量级；
     merge_summary 应显示 `src≈59 na≈0`；若 `[NO-SRC]` 整列 N/A → 先 `grep -c
     '^perf,' build/benchdnn_*.txt` 确认确有 perf 行（--mode=p 应有），并确认
     merge 是最新版（旧 merge 曾因缩写 prb 缺 iw 误报 [NO-SRC]，已修）。
   - **P0/P1 频探**：主频应稳定 2.0GHz（606/608 核），漂移说明节点状态异常。

1. **同作业才可比**：任何跨作业数字（含本文历史数字）只作定性参考，性能结论必须
   `sbatch -w node03 --exclusive` 单作业内 A/B。
2. **ours 侧无 debug 才可信**：`WINO_GEMM_DEBUG=1`（E1）会不成比例放大小形状计时，
   新旧/大小对比只信 E3/E5 无 debug 读数。
3. **修正旧文档口径**：README/AGENTS 里「8/9 case 超越 oneDNN」「6/9 超越」是
   **OpenBLAS 后端 + benchdnn WINO 口径**的历史数字（oneDNN 被测量方式拖慢 1.7~3.6x），
   不能当 arm_gemm 后端的结论；arm_gemm 后端 vs OpenBLAS 才是 59/59 终局。
4. **oneDNN WINO 依赖 ACL**：本原生 build（无 ACL）可能全 unimplemented → wino 列全
   N/A，以 auto 列为主。
5. **git 边界**：不要 stage 用户未提交的 `README.md` 与 `swish_sve/`；`tools/diag_ab.sh`
   是临时诊断脚本，闭环后删除。
