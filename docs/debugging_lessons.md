# 排障经验：oneDNN e2e 系统性 OOM 的弯路与终点

> 记录一次修了两轮、第一轮修错的 bug。**根因与判读矩阵见 `docs/onednn_comparison.md` §四**；
> 本文写的是**弯路本身**——为什么第一理论看起来对、怎么被证伪、有哪些可复用的方法论教训。
> 目的：让排障更快，避免把"说得通的解释"当成"证据"。

## TL;DR

- **症状**：oneDNN e2e 里每个 conv 的 primitive_desc 创建都 `out_of_memory`，59 形状全挂。
- **第一理论（错）**：`omp_set_num_threads` 触发 → 删掉该调用（8244944）→ 集群实测全无效。
- **真根因**：给**无膨胀** conv 显式传了 `dilates={1,1}`（应为 {0,0}），错误 dilation 直达
  集群 lib 的 impl → 巨大分配 → 系统性 OOM。修复 = 改用「带 bias、不带 dilates」的公开重载（4bdc3ee）。
- **一句话教训**：**一个"说得通"的解释不是证据；证据来自能区分两个假设的对照实验。**

## 一、完整时间线

### 阶段 0：症状与第一印象

e2e 输出：`[preOMP] ok (impl=jit:sve)`，然后 `[smoke] FAIL (oom)`，再然后所有 59 个 conv 的
PD 创建 `status=2(out_of_memory)`。第一印象：**"第 1 个 primitive 能建、后面全炸"** → 很像
"进程内状态被污染 / 线程数把库搞坏" → 联想到初版代码里确实有一个 `omp_set_num_threads` 调用。

### 阶段 1：错修 8244944

理论自洽：oneDNN 文档确实警告过外部线程数与内部 OpenMP 冲突；调用就摆在代码里。于是删掉它，
更新了 memory 与脚本注释，宣布"已修复"。**没有先在集群验证。**

### 阶段 2：被证伪

集群实测三组配置，[smoke] **全部依旧 oom**：

| 配置 | 变化 | 结果 |
|---|---|---|
| 正常（OMP_NUM_THREADS=16） | 基线 | [smoke] FAIL |
| nothr（无 env，库默认 = 608 线程） | 去掉线程约束 | [smoke] FAIL |
| nosve（ONEDNN_MAX_CPU_ISA=advanced_simd） | 关掉 SVE，impl 退 jit:asimd | [smoke] FAIL |

一个变量（omp 调用/线程数）变化而症状纹丝不动 ⇒ **该变量不是根因**。8244944 理论死亡。

### 阶段 3：红鲱鱼排除

`[preOMP] ok / [smoke] FAIL` 曾被视为"第 1 个 PD 好、第 2 个中毒"。对照代码后发现两者
**格式不同**：preOMP = eltwise-nchw（具体格式），smoke = eltwise-**any**（format_any）。
本地 3.12.1 源码 `eltwise.cpp` 直接 reject `format_any` 的前向 eltwise —— 该差异是**格式**
导致，不是顺序/中毒。eltwise 分支全部排除出 conv 根因。

### 阶段 4：本地源码与集群实测的矛盾

本地下载的 3.12.1 源码 `convolution.cpp:166` 是 `VCHECK_CONV → invalid_arguments`（status 1），
集群实测却是 `out_of_memory`（status 2）。**两者矛盾 ⇒ 集群 lib ≠ 本地源码**。此时正确的
态度是：**一切以设备实测为准**，不再为源码行为找解释。

### 阶段 5：用户参考程序 = 地面真值

用户给出一个**在集群上成功跑过的**参考程序的 conv PD 构造格式——带 bias、**不带 dilates**
的重载。这是"在这个 lib 上真的能建 PD"的事实样本，比任何源码推理都可靠。采纳之：
`dnnl.hpp` 该重载内部把 dilates 置 0，等于自动修正了 `{1,1}` 的错误。

### 阶段 6：修复 + 诊断矩阵

4bdc3ee：conv PD 一律走无 dilates 重载；bias 恒用具体 `tag::x`；算法梯子 direct+fwdinf 放首位
（= 参考程序 proven 组合）；探针移到数据流之后；新增 `--diag` 单 PD 诊断矩阵与逐 shape 独立
进程兜底。诊断矩阵一行一个独立进程，**一个进程只建一个 PD**，彻底消除跨 PD 状态干扰，
让"根因是否已除"可被单变量验证：`conv_direct_any_nodil=ok` 且 `conv_direct_any_dil1=FAIL`。

## 二、方法论教训（可复用清单）

1. **"说得通的解释" ≠ 证据。** 修完必须设备验证；验证前不要更新 memory/文档宣布修复
   （8244944 的错误就记进了 memory，差点变成永久错误基线）。
2. **给每个假设设计杀手实验。** 对照要**只改一个变量**：nothr / SVE-off 两配置各只改一件事，
   症状不变 ⇒ 理论淘汰。别在同一个配置里同时改两个变量（会无法归因）。
3. **"现象解释"可能是巧合。** [preOMP]ok/[smoke]FAIL 看起来像"顺序/中毒"，实际是"格式"差异。
   下次遇到"第 1 个好第 2 个坏"，先问：**还有哪个变量在两个 case 之间变了？**
4. **本地源码 ≠ 设备库。** 源码推理与设备观察矛盾时，设备实测优先。源码只能辅助理解，
   不能推翻实测。
5. **用户的参考程序是地面真值。** 有"在目标环境成功跑过的调用格式"，直接采纳/对照，
   比自己推导 API 语义快且稳。
6. **每个传参都对数据本身。** CSV 说 dh=dw=0（无膨胀），代码却传 dilates={1,1}。
   写参数时对照输入数据，别信"没人用这个字段"。
7. **诊断要能单变量复现**：一行一个独立进程、只建一个 PD，才能干净地回答"根因除没除"。

## 三、下个 bug 的 SOP（省时版）

1. **复现 + 最小化**：拿到集群失败输出，先分清"哪个调用在哪个参数下失败"。
2. **列假设 → 杀手实验**：把"在变化的变量"列全，对每个假设设计只改它的对照。
3. **设备实测为准**：源码只作辅助；实测与源码矛盾时信实测。
4. **修完四步闭环**：正确性门 → 数据流出数 → 诊断矩阵判读 → **才**更新文档/memory。
5. **参考用户已知可跑的样本**：有成功的参考调用格式就先对齐它。

## 四、benchdnn [NO-SRC] 弯路（2026-08-31）

> oneDNN 对照的 benchdnn 列从 [NO-SRC] 到 src=59 的排障复盘。6 条弯路，每条都有
> "看起来对但实际错"的第一理论。根因与最终方案见 `docs/onednn_comparison.md` §六。

### 弯路 1：误判 CRLF（2 轮）

**症状**：`bash tools/onednn/run_benchdnn.sh --winograd --threads 16` 执行后零输出，
文件 mtime 不变（仍 Aug 29）。

**第一理论**：SFTP 从 Windows 传文件带 `\r\n`，`set -euo pipefail` 下 `cd "path\r"`
路径含 `\r` 失败 → 脚本静默退出。

**证伪**：用户 `sed -i 's/\r$//'` 剥掉 CRLF 后重跑——仍然零输出。CRLF 不是根因。

**真根因**：参数解析 `esac` 后有多余的 `shift`（line 39）。`--threads 16` 在 case
里 `shift 2` 消费两个参数后，外面 `shift` 对空参数返回 1 → `set -e` 静默退出。
**`shift` 在 `set -e` 下对空参数返回 1 是静默的（无错误消息）**——和 CRLF 的
"静默退出"表现一模一样，导致误判。

**教训**：`bash -x script.sh` 能一下看到脚本在哪行退出。**set -e 的静默退出
不提供任何错误信息，必须用 -x trace 定位。**

### 弯路 2：`--mode=p` 不出 perf 行（2 轮）

**症状**：集群 benchdnn `--mode=p` 跑完后 `grep -c '^perf,'` = 0。

**第一理论**：该 oneDNN 3.12.1 build 不支持 `--mode=p` 的 perf 输出格式。

**证伪**：加 `ONEDNN_VERBOSE=exec` env var → 仍 0 exec 行。加 PASSED 行解析兜底
→ 出了 59 行但计时是聚合时间（含 fill/ref/compare），不可比。

**真根因**：oneDNN 3.12.1 需要 `ONEDNN_VERBOSE=1` env var 才启用任何 verbose
输出。`-v4` flag 只在 `ONEDNN_VERBOSE=1` 已设的前提下才生效。用户的本地
oneDNN 3.4.0 不需要 env var（`-v4` 自动启用），导致版本差异被忽略。

**教训**：**env var 的优先级和值因 oneDNN 版本而异**。3.4.0 自动启用 verbose，
3.12.1 要求 `ONEDNN_VERBOSE=1`。跨版本行为差异不能用一个版本的观察推断另一个。

### 弯路 3：e2e 0 行 + ldd 无 dnnl（2 轮）

**症状**：`onednn_e2e.csv: 0 rows`，`ldd build/onednn_e2e` 显示无 dnnl/omp/tbb。

**第一理论**：编译失败 / ROOT 探测选错路径 / CRLF 导致编译命令含 `\r`。

**证伪**：加诊断打印——ROOT/LIBS_DIR/CXX 值正确，`libdnnl.so` 存在于 LIBS_DIR，
编译 exit=0。但 ldd 仍无 dnnl。

**真根因**：oneDNN release 库链接 `libarm_compute.so`（ACL），但 ACL 库不在
`LD_LIBRARY_PATH`。编译时 `-ldnnl` 找到了（dnnl 依赖 ACL），但运行时 ACL 库
找不到 → 二进制启动即崩溃 → 0 行输出。用户手动加 ACL 路径到 `LD_LIBRARY_PATH`
解决。

**教训**：**ldd 是诊断链接问题的第一步**。`ldd binary | grep dnnl` 一看就知道
是否链接成功。不要在没看 ldd 的情况下猜编译失败。

### 弯路 4：`grep -c || echo 0` 产生 `0\n0`（1 轮）

**症状**：自检行 `perf_lines=0\n0 exec_lines=0\n0`，`[: 0\n0: integer expression
expected`。

**根因**：`grep -c` 无匹配时打印 `0` 到 stdout 且返回 exit code 1。`|| echo 0`
再打印 `0`。`$(...)` 捕获到两行 `0\n0`。

**修复**：`VAR=$(grep -c ...) || VAR=0`——`||` 后赋值不产生 stdout。

**教训**：**`grep -c` 无匹配时 exit code = 1，但不报错——和"文件不存在"不可区分**。
用 `|| VAR=0`（不是 `|| echo 0`）避免 stdout 污染。

### 弯路 5：ONEDNN_VERBOSE=1 污染 e2e CSV（1 轮）

**症状**：e2e 输出混入 `onednn_verbose,v1,...` 行，CSV 被污染。

**根因**：用户 shell 设了 `export ONEDNN_VERBOSE=1`（为 benchdnn 调试），sbatch
作业继承该 env var。e2e 的 oneDNN 库看到 `ONEDNN_VERBOSE=1` 就打印 verbose
行到 stdout（混入 CSV）。

**修复**：e2e 主运行加 `ONEDNN_VERBOSE=0`（探针用 `=all`，独立进程不受影响）。

**教训**：**export 的 env var 会污染同作业内所有子进程**。调试完后要 unset，或
在脚本里显式覆盖。

### 弯路 6：nthr:608 担忧（1 轮）

**症状**：手动测试 benchdnn 显示 `nthr:608`（608 线程），担心计时被 608 线程
超订放大。

**修复**：加 `DNNL_NUM_THREADS=16` + `OMP_THREAD_LIMIT=16` env var。集群实测
自检行 `nthr=16`——生效了。

**教训**：`OMP_NUM_THREADS` 对 oneDNN 3.12.1 + bisheng libomp **可能不生效**
（e2e 通过 `omp_set_num_threads` API 调用生效，benchdnn 不调用 API 只靠 env var）。
`DNNL_NUM_THREADS` 是 oneDNN 自己的 env var，比 `OMP_NUM_THREADS` 更可靠。

### 弯路总结：方法论教训

1. **`bash -x` 是定位 set -e 静默退出的唯一利器**——shift bug 用 -x 一下就能看到。
2. **ldd 是诊断链接问题的第一步**——不要在没看 ldd 的情况下猜编译失败。
3. **env var 的值和优先级因版本而异**——`ONEDNN_VERBOSE=1` vs `=exec` vs `=all`
   在 3.4.0 和 3.12.1 上行为不同，不能用一个版本的观察推断另一个。
4. **`grep -c` 的 exit code 陷阱**——无匹配返回 1，用 `|| VAR=0` 不用 `|| echo 0`。
5. **export 的 env var 会污染同作业子进程**——调试完要 unset 或脚本里显式覆盖。
6. **"静默退出"有多种原因**——CRLF、set -e + shift、链接缺失，表现一模一样（零输出），
   必须 `bash -x` + `ldd` + `stat` 三板斧快速定位。
