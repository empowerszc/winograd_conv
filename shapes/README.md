# shapes/ —— 与 oneDNN（benchdnn）可对照的基准用例集

`bench_winograd` 的性能数据要与用户已有的 oneDNN benchdnn 数据做**同 shape 对照**：两边跑
完全相同的 `mb,ic,ih,iw,oc`，比较时间差异与 profiling 差异（分析文档见
`docs/why_faster_than_acl_23.11.md`）。

本目录刻意**不**追求 benchdnn 格式兼容（不生成 `.dnn`、不解析 benchdnn 输出）。数据创建与
测量全部走本仓库自己的 `bench_winograd`：NHWC、fp32、F(4,4,3,3)、best-of-repeats。

## 文件格式

`conv_all.csv` 为本仓库原生 CSV，全部行满足 `bench_winograd` 的过滤条件：
**stride=1、group=1、3x3 核、pad=1**（其它参数会被 `read_shapes` 标为 skipped）。

```text
mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count
```

- `#` 开头的行是节标题/注释，解析时跳过（`bench_winograd.cpp` 的 `read_shapes` 已支持）。
- `count` 列只作信息记录（对应 benchdnn 的 `n` 迭代参数）；我们侧计时由 `--repeats` 控制。
- 首行即上述表头（`read_shapes` 无条件跳过首行）。

## 各节测哪个假设

| 节 | 行数 | 覆盖假设（why_faster §） | 说明 |
|---|---|---|---|
| `# Case 0-8` | 9 | §2.1 历史 9 case | 与之前 F44/F22 数据直接接轨 |
| `# ResNet50 3x3 s1` | 12 | §4.1 固定开销 | 真实 CNN，mb=1 单图（小负载）+ mb=4；含 4× 降维 bottleneck 与 2048 深通道 |
| `# VGG16 3x3 s1` | 10 | §4.1 / §8.5 | 224² 大图小通道、14² 深通道，mb=1/4 两组 |
| `# Lightweight / small-image` | 4 | §3.1 小负载 | 轻量网络典型形状，验证小调用胜负 |
| `# Large batch` | 4 | §4.1 / §8.5 | mb=8~32 摊薄固定开销——若大负载仍赢则说明是系统性优势而非固定开销 |
| `# Fixed-overhead isolation` | 4 | §3.1 / §4.1 | 最小负载 + 高 count，隔离固定开销与调度成本 |
| `# tiles x IC sweep grid` | 16 | §8.7 热力图 / §3.2 | mb=4、OC=IC，tiles∈{25,100,400,1600}（IH=IW∈{20,40,80,160}）× IC∈{48,96,192,384} 全组合 16 点 |

合计 59 行。tiles×IC 网格与 Case 0-8 有 4 个格点重叠——有意为之，便于热力图与历史数据
对照。

## 怎么跑

### 我们侧（920F）

先正确性门，再计时：

```bash
# 逐 case 正确性：fp64 直接卷积参考，相对容差 1e-4，FAIL 中止
./build/bench_winograd --sve --nhwc --verify --threads 16 shapes/conv_all.csv

# 计时（best-of-repeats）
./build/bench_winograd --sve --nhwc --threads 1,8,16,32,38 shapes/conv_all.csv

# 细粒度 per-step 计时（transform / GEMM / 写回）
./build/bench_winograd --sve --nhwc --timing --threads 16 shapes/conv_all.csv
```

### 极简对比脚本

```bash
./tools/compare.sh --threads 16 --isa sve shapes/conv_all.csv
```

输出紧凑的 `mb,ic,ih,iw,oc,ours_ms` 表（可直接贴进与 oneDNN 数字的对照表），首行 `# run:`
记录本次参数；**全量**原始结果落在 `build/compare_ours.csv`。脚本不解析 benchdnn 输出。

### oneDNN 侧（benchdnn，按你自己的方式跑）

同一行 shape 的 benchdnn 描述符：`pad1/s1/k3` ⇒ `oh=ih, ow=iw`。规则：

```text
CSV 行  4,192,40,40,192 → 描述符  mb4_ic192_ih40iw40_oc192_oh40ow40_kh3kw3_sh1sw1_ph1pw1
```

即 `mb<mb>_ic<ic>_ih<ih>iw<iw>_oc<oc>_oh<oh>ow<ow>_kh3kw3_sh1sw1_ph1pw1`。例子：

```bash
./benchdnn --conv --cfg=f32 --reset --alg=WINO --batch=conv_all.list
# conv_all.list 每行：
#   mb4_ic192_ih40iw40_oc192_oh40ow40_kh3kw3_sh1sw1_ph1pw1_n"case0"
```

## profiling 对照（两边跑同一批计数器）

920F **无 L3**，`LLC-*` 事件不存在，用 L1/L2 事件。计数器清单与命令见
`docs/why_faster_than_acl_23.11.md` §8.3（perf）/ §8.4（SPE）。常用：

```bash
# 基础：周期/指令/IPC/分支
perf stat -e task-clock,cycles,instructions,branches,branch-misses \
  ./build/profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --warmup 5 --repeats 1

# cache（无 L3 → L1/L2）
perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,\
  L2-dcache-loads,L2-dcache-load-misses ./build/profile_case ...

# topdown / NUMA
perf stat --topdown ./build/profile_case ...
perf stat --per-node ./build/profile_case ... && numastat

# SPE 访存局部性采样
perf record -e arm_spe_0/ts_enable=1,pa_enable=0,min_latency=0,period=10000/ \
  -- ./build/profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --warmup 5 --repeats 1
```

对同一 `mb,ic,ih,iw,oc` 的 shape，两边各取一份上述输出即可比较：总指令数、
L2 miss 率、topdown 桶、访存地址分布（连续段 vs 跨步）。
