# docs/figures —— 汇报用静态图示（HTML）

面向汇报 / 答辩 / 打印的一套独立网页，**每页只讲一件事**，数字全部可追溯到仓库里的
markdown 表。全部是自包含 HTML（内联 CSS + 内联 SVG），**无 CDN、无 JS 库、无外部字体**，
拷到任何机器双击即可打开，离线可用。

生成入口：`tools/gen_report_figures.py`（不手改 HTML）。

## 页面

| 文件 | 内容 | 主要数据源 |
|---|---|---|
| `index.html` | 总览：四个关键指标 + 图示索引 + 测量口径速查 | 本目录 |
| `01_winograd_principles.html` | tile 划分与重叠、三段流水、144→36 乘法、Bᵀ 零系数 | `docs/algorithm.md` |
| `02_system_architecture.html` | NHWC 核心 + NCHW 包装、三级 ISA、GEMM 后端、单 OpenMP 区域 | `src/winograd_conv.cpp` |
| `03_sme_gemm.html` | ZA 布局、drain 切片、微内核汇编、`smstart` clobber 陷阱 | `docs/sme_fmopa_gemm.md` |
| `04_performance.html` | 59 形状逐条加速比、分布、四种平均、SME vs KML 两档 | `docs/final_benchmark_bfd6b1e.md`、`docs/sme_fmopa_gemm.md` |
| `05_acl_comparison.html` | 与 ACL 的同源清单 / 八层差异 / 优势与差距 | `docs/vs_acl.md` |
| `06_timing.html` | 8 步耗时占比、三种形态、小 GEMM 崩塌、并行效率 | `docs/timing_breakdown_920f.md` |
| `07_methodology.html` | 同作业 A/B、绑核、四种平均、方向约定、已知陷阱 | 各文档口径章节 |
| `08_roadmap.html` | 三阶段九项任务、判定标准、四条已证伪方向 | `docs/project_summary.md` §9 |
| `09_bench_shapes.html` | 59 形状的七个小节、tiles × IC 网格热力图 | `shapes/conv_all.csv`、`shapes/README.md` |

## 重新生成

```bash
python3 tools/gen_report_figures.py
```

脚本会解析两份 markdown 的表格（`final_benchmark_bfd6b1e.md` 的 59 行加速比、
`sme_fmopa_gemm.md` §5.3/§5.4 的 59 行 SME/KML，以及 `timing_breakdown_920f.md`
的 §3.1 与附录 A），所以**改了文档数据，重跑脚本即可同步图示**，不会出现图文不一致。

输出末尾会打印口径自检：

```
summary: ob speedup=2.501 (ratio 0.400) geo=1.992 arith=2.171 median=1.771
summary: sme t1 ratio=0.975 speedup=1.026 geo=1.007 arith=1.060 median=1.014 wins=31
summary: sme t16 ratio=0.896 speedup=1.117 geo=0.901 arith=0.989 median=1.021 wins=32
```

若解析行数不是 59（或 §3.1 不是 8），脚本会打印 warning 到 stderr——说明上游表格被改动。

## 阅读约定

- **比值的分子分母**：`vs OpenBLAS` 用 `OpenBLAS / 本项目`（>1 = 我们快）；
  `SME vs KML` 的合计用 `SME / KML`（<1 = SME 快），逐形状柱高用 `KML / SME`（>1 = SME 快）。
- **集群与本机不可跨机比较**：04/09 页的数字来自集群 920F；03/04 页的 SME 行来自本机 608 核。
- **图表可悬停**：SVG 里每个柱/点/格子都带 `<title>`，鼠标停留可看到该形状的原始毫秒数。
- **打印**：`@media print` 已隐藏导航、去掉阴影，Cmd/Ctrl+P 可直接出 PDF（建议 A4 横向 + 背景图形）。

## 已知边界

- 无外部字体：中文依赖阅读机器自带字体（Windows/macOS/Linux 常见字体栈已列出）。
- 页面无交互脚本：排序/筛选不做，信息靠分节与图例表达。
- oneDNN 逐形状数据不在仓库（只在集群 `build/`），所以 04 页只能给出区间。
