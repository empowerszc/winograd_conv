// oneDNN 端到端 fp32 3x3 conv 基准 —— 与 bench_winograd 同 shape(CSV) 对照。
//
// 公平口径（docs/why_faster_than_acl_23.11.md §10.4）：
//   - primitive 跨迭代复用（不重建、权重变换由 oneDNN 内部缓存）
//   - src/dst/weights/bias 全用 format_any，让 oneDNN 自选布局与算法
//   - 线程绑定交给外层脚本（OMP_PROC_BIND=close OMP_PLACES=cores）
// 计时语义与 compare.sh 对齐：warmup 后 repeats 次取 MIN（best-of-repeats）。
//
// 构建（脚本里自动做，等价于）：
//   $CXX -O3 -std=c++17 -fopenmp -I$ONEDNN_ROOT/include onednn_e2e.cpp \
//        -L$ONEDNN_ROOT/lib -Wl,-rpath,$ONEDNN_ROOT/lib -ldnnl -o build/onednn_e2e
// 用法：
//   onednn_e2e <shapes.csv> <threads> [warmup] [repeats] [--auto|--winograd]
// 输出：
//   # run: threads=16 warmup=3 repeats=20 alg=auto
//   mb,ic,ih,iw,oc,onednn_ms
//   （stdout 仅数据行；诊断/进度走 stderr）
//
// 算法梯子：oneDNN 3.12.1 原生 AArch64 build 的 conv auto 路径在 PD 创建阶段
// 系统性返回 out_of_memory（与 shape/布局/prop_kind 无关，见 run 输出）。本程序对
// 每个 shape 依次试 {auto, direct, winograd}（--winograd 则 winograd 放首位），
// 第一个能建成 PD 的算法获胜；每行数据在 stderr 标注 via alg=X pk=Y impl=Z layout=W，
// 其中 impl=Z 来自 pd.impl_info_str()（dnnl::primitive_desc 基类成员，3.12.1 存在）
// ——直接显示实际选中的实现名（如 brgconv:sve_512 / gemm:f32），是 OOM 定位的关键
// 证据。若全部算法都失败，stderr 打一行紧凑探针 [PD-ALL]，含每个 alg/pk 的数值状态码，
// 用于定位是哪个实现族坏。注意 dnnl::algorithm 只暴露这三个成员（无
// convolution_gemm，3.12.1 C++ 包装收敛过）；2.x 时代的
// convolution_forward::desc + primitive_desc(desc, eng) 构造形式在 3.0 已移除，
// 3.x 只有 engine 开头的一串直接构造，本文件用的正是该形式。
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Shape {
    int mb, ic, ih, iw, oc;
    int kh, kw, sh, sw, ph, pw, dh, dw, grp;
};

bool parse_csv(const std::string& path, std::vector<Shape>& out) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path.c_str()); return false; }
    char line[512];
    bool first = true;
    while (fgets(line, sizeof line, f)) {
        if (first) { first = false; continue; }       // 表头（同 read_shapes）
        if (line[0] == '#') continue;                 // 节标题/注释
        Shape s;
        if (sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                   &s.mb, &s.ic, &s.ih, &s.iw, &s.oc, &s.kh, &s.kw,
                   &s.sh, &s.sw, &s.ph, &s.pw, &s.dh, &s.dw, &s.grp) == 14) {
            if (s.kh == 3 && s.kw == 3 && s.sh == 1 && s.sw == 1 &&
                s.ph == 1 && s.pw == 1 && s.dh == 0 && s.dw == 0 && s.grp == 1)
                out.push_back(s);
        }
    }
    fclose(f);
    return true;
}

void fill(float* p, size_t n) {   // 小幅度伪随机，避免 inf/nan
    for (size_t i = 0; i < n; i++)
        p[i] = float(int((i * 2654435761u) % 17u) - 8) * 0.01f;
}

const char* status_name(int st) {
    switch (st) {
        case 0:  return "success";
        case 1:  return "invalid_arguments";
        case 2:  return "out_of_memory";
        case 3:  return "unimplemented";
        case 4:  return "iterator_ends";
        case 5:  return "runtime_error";
        case 6:  return "not_required";
        case 9:  return "invalid_shape";
        case 10: return "invalid_data_type";
        default: return "unknown";
    }
}

const char* st_short(int st) {   // 紧凑状态缩写，供 [PD-ALL] 探针行（一行放 8 个组合）
    switch (st) {
        case 0:  return "ok";
        case 1:  return "inv";
        case 2:  return "oom";
        case 3:  return "uni";
        case 4:  return "end";
        case 5:  return "rte";
        case 6:  return "nreq";
        case 9:  return "shp";
        case 10: return "dt";
        default: return "?";
    }
}

// 编码无关的错误 dump：可打印 ASCII 原文 + 权威十六进制字节流。集群终端可能
// 是 GBK、错误文本可能含非 UTF-8 字节——只打印 what() 会变乱码，hex 永不失真。
void dump_err(FILE* f, const char* stage, const Shape& s, int status,
              const std::string& msg) {
    fprintf(f, "skip %d,%d,%d,%d,%d [%s]: status=%d(%s) what=",
            s.mb, s.ic, s.ih, s.iw, s.oc, stage, status, status_name(status));
    for (unsigned char c : msg) fprintf(f, "%c", (c >= 32 && c < 127) ? (char)c : '.');
    fprintf(f, " hex=");
    for (unsigned char c : msg) fprintf(f, "%02x", (unsigned)c);
    fprintf(f, "\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <shapes.csv> <threads> [warmup] [repeats] [--auto|--winograd]\n", argv[0]);
        return 1;
    }
    const char* csv_path = argv[1];
    int threads = atoi(argv[2]);
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    int repeats = argc > 4 ? atoi(argv[4]) : 20;
    std::string alg_s = "auto";
    for (int i = 5; i < argc; i++)
        if (std::string(argv[i]) == "--winograd") alg_s = "winograd";
        else if (std::string(argv[i]) == "--auto") alg_s = "auto";

    std::vector<Shape> shapes;
    if (!parse_csv(csv_path, shapes)) return 1;
    fprintf(stderr, "parsed %zu shapes\n", shapes.size());   // 自查：0 => parse 失败

    // verbose 1 级：dnnl_verbose,info 行（runtime/nthr/isa）打进 stderr，
    // 随主流程 stderr 一起输出，确认库对自己运行环境的检测结果。
    dnnl::set_verbose(1);

    // [preOMP] 在设置任何线程数/绑定之前，先用 C++ 包装建一个 eltwise PD：
    // 此刻无 OMP 线程池、无任何 oneDNN 调用。若成功而后面 [smoke] oom ⇒ OMP
    // 线程池/绑核/env 是触发器；若同样 oom ⇒ 库本身在此环境就坏（换安装）。
    {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::memory::dims d{1, 16, 16, 16};
        auto smd = dnnl::memory::desc(d, dt::f32, tag::nchw);
        dnnl::engine e0(dnnl::engine::kind::cpu, 0);
        try {
            auto epd = dnnl::eltwise_forward::primitive_desc(e0,
                    dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu,
                    smd, smd, 0.0f, 0.0f);
            fprintf(stderr, "[preOMP] eltwise PD ok impl=%s\n",
                    epd.impl_info_str());
        } catch (const dnnl::error &e) {
            fprintf(stderr, "[preOMP] eltwise PD FAIL status=%d(%s) what=%s\n",
                    (int)e.status, status_name((int)e.status), e.what());
        }
    }

    // 线程数不调 omp_set_num_threads（2026-08-29 修复）：3.12.1 本 build 在
    // OMP_PROC_BIND=close OMP_PLACES=cores 下，进程内调用它会让此后所有 oneDNN
    // PD 创建系统性返回 out_of_memory——[preOMP] 探针在它之前成功（impl=jit:sve），
    // [smoke] 在它之后 oom，而 benchdnn 同 env 不调该函数则 59 个 conv PD 全过，
    // ⇒ 触发器就是这次调用。线程数改由外层脚本的 OMP_NUM_THREADS env 提供
    // （libomp 首次初始化即读取，无 API 调用），与 benchdnn 同口径。
    // dnnl::set_max_threads 也不可用：它随 DNNL_CPU_THREADING_RUNTIME 条件编译，
    // 部分发行包（如 3.12.1-release）里根本不存在。
#ifdef _OPENMP
    fprintf(stderr, "[thr] omp_get_max_threads=%d omp_get_num_procs=%d\n",
            omp_get_max_threads(), omp_get_num_procs());
#endif
    // 算法梯子：oneDNN 3.12.1 的 C++ 枚举只暴露 {auto, direct, winograd}
    // （dnnl::algorithm 无 convolution_gemm——该实现族由 direct 覆盖）。
    // 首个能建成 PD 的获胜；本 build 无 ACL 故 winograd 大概率 unimplemented。
    std::vector<dnnl::algorithm> ladder;
    std::vector<const char*> ladder_name;
    if (alg_s == "winograd") {
        ladder = { dnnl::algorithm::convolution_winograd,
                   dnnl::algorithm::convolution_direct,
                   dnnl::algorithm::convolution_auto };
        ladder_name = { "winograd", "direct", "auto" };
    } else {
        ladder = { dnnl::algorithm::convolution_auto,
                   dnnl::algorithm::convolution_direct,
                   dnnl::algorithm::convolution_winograd };
        ladder_name = { "auto", "direct", "winograd" };
    }

    // ---- 库级冒烟探针（PD 创建系统性 OOM 时二分用）----
    // 源码分析（primitive_desc_iface.cpp:73-74）：out_of_memory 只可能来自
    //  ① pd_iterator_ 为空（分配失败） ② pd_iterator_->is_initialized() 为假
    // （attr_.is_initialized() 被读错——header/lib 结构体布局不一致的 ABI 症状）。
    // 实现级 create 失败只会被迭代器跳过 → 最终 unimplemented，绝不会 oom。
    // 所以只要 PD 创建 oom，就与算法/形状/布局/ISA 无关，是库级故障（集群观察到的
    // 全 oom、SVE-off 也 oom、winograd 空实现列表也 oom 与此完全吻合）。
    // ① header 宏 vs 链接库运行时版本：版本不一致 = ABI 破坏，该包直接判废。
    const auto *ver = dnnl::version();
    fprintf(stderr,
            "[env] header DNNL_VERSION=%d.%d.%d lib runtime=%d.%d.%d "
            "cpu_runtime=%u gpu_runtime=%u hash=%s\n",
            DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
            ver->major, ver->minor, ver->patch, ver->cpu_runtime,
            ver->gpu_runtime, ver->hash ? ver->hash : "(null)");
    // ② eltwise PD：非 conv 原语，验证引擎/attr/迭代器整条建 PD 路径。
    //    若 eltwise 也 oom ⇒ 库二进制级坏（换 oneDNN 安装/重编）；若 eltwise
    //    正常而 conv 全 oom ⇒ conv 专属（罕见，按 conv 实现表继续查）。
    {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::memory::dims d{1, 16, 16, 16};
        auto smd = dnnl::memory::desc(d, dt::f32, tag::any);
        dnnl::engine e0(dnnl::engine::kind::cpu, 0);
        // ②a C++ 包装 + 默认 attr（与主流程同路径）
        try {
            auto epd = dnnl::eltwise_forward::primitive_desc(e0,
                    dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu,
                    smd, smd, 0.0f, 0.0f);
            fprintf(stderr, "[smoke] eltwise PD ok impl=%s\n",
                    epd.impl_info_str());
        } catch (const dnnl::error &e) {
            fprintf(stderr, "[smoke] eltwise PD FAIL status=%d(%s) what=%s\n",
                    (int)e.status, status_name((int)e.status), e.what());
        }
        // ②b 原始 C API + attr=NULL：排除 C++ 包装/default_attr 的影响。
        //    若这里成功而 ②a oom ⇒ 问题在 C++ 包装的 attr 路径（可绕行全用
        //    C API + null attr）；若这里也 oom ⇒ 库 PD 机器层整体坏，只能换装。
        {
            dnnl_primitive_desc_t cpd = nullptr;
            dnnl_status_t cst = dnnl_eltwise_forward_primitive_desc_create(
                    &cpd, e0.get(), dnnl_forward_training, dnnl_eltwise_relu,
                    smd.get(), smd.get(), 0.0f, 0.0f, nullptr);
            fprintf(stderr,
                    "[capi-eltwise] attr=NULL PD create=%d(%s)\n",
                    (int)cst, status_name((int)cst));
            if (cst == dnnl_success && cpd) dnnl_primitive_desc_destroy(cpd);
        }
        // ②c 最小分配路径 + 进程堆：区分「节点内存/cgroup 真不足」vs「库内故障」
        {
            dnnl_primitive_attr_t ca = nullptr;
            dnnl_status_t st = dnnl_primitive_attr_create(&ca);
            fprintf(stderr, "[capi] dnnl_primitive_attr_create=%d(%s)\n",
                    (int)st, status_name((int)st));
            if (st == dnnl_success && ca) dnnl_primitive_attr_destroy(ca);
        }
        {
            void *p = malloc(256u << 20);
            fprintf(stderr, "[heap] malloc(256MB)=%s\n", p ? "ok" : "FAIL");
            free(p);
        }
    }

    fprintf(stdout, "# run: threads=%d warmup=%d repeats=%d alg=%s\n",
            threads, warmup, repeats, alg_s.c_str());
    fprintf(stdout, "mb,ic,ih,iw,oc,onednn_ms\n");

    bool first_fail_detail = true;   // 首个 [PD-ALL] 形状额外打一次完整 hex 消息
    for (const auto& s : shapes) {
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream stream(eng);

        using md = dnnl::memory::desc;
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::memory::dims src_d{ s.mb, s.ic, s.ih, s.iw };
        dnnl::memory::dims wei_d{ s.oc, s.ic, 3, 3 };
        dnnl::memory::dims bia_d{ s.oc };
        dnnl::memory::dims dst_d{ s.mb, s.oc, s.ih, s.iw };
        dnnl::memory::dims stride{ 1, 1 }, dilate{ 1, 1 }, padl{ 1, 1 }, padr{ 1, 1 };

        // 单形状完整跑一遍（可指定布局）。任一步失败 → dump【阶段 + 数值状态码 +
        // hex 字节】并返回 false（不依赖终端编码即可定位；之前只打 e.what() 在 GBK
        // 终端是乱码）。成功打印一行数据并返回 true。
        auto run_one = [&](const char* layout) -> bool {
            md s_md, w_md, b_md, d_md;
            if (std::strcmp(layout, "nchw") == 0) {   // 显式布局：强制 ref/gemm 路径
                s_md = md(src_d, dt::f32, tag::nchw);
                w_md = md(wei_d, dt::f32, tag::oihw);
                b_md = md(bia_d, dt::f32, tag::x);
                d_md = md(dst_d, dt::f32, tag::nchw);
            } else {                                  // format_tag::any：oneDNN 自选
                s_md = md(src_d, dt::f32, tag::any);
                w_md = md(wei_d, dt::f32, tag::any);
                b_md = md(bia_d, dt::f32, tag::any);
                d_md = md(dst_d, dt::f32, tag::any);
            }

            dnnl::convolution_forward::primitive_desc pd;
            bool have_pd = false;
            const char* via_alg = "";
            const char* via_pk = "";
            std::string combos;         // 失败组合摘要 "alg/pk=st_short "（编码无关探针）
            int last_status = 0;
            std::string last_msg;
            for (size_t ai = 0; ai < ladder.size() && !have_pd; ai++) {
                for (auto pk : { dnnl::prop_kind::forward, dnnl::prop_kind::forward_inference }) {
                    const char* pkn = (pk == dnnl::prop_kind::forward) ? "fwd" : "fwdinf";
                    try {
                        pd = dnnl::convolution_forward::primitive_desc(
                            eng, pk, ladder[ai], s_md, w_md, b_md, d_md,
                            stride, dilate, padl, padr);
                        have_pd = true; via_alg = ladder_name[ai]; via_pk = pkn;
                        break;
                    } catch (const dnnl::error& e) {
                        char buf[96];
                        snprintf(buf, sizeof buf, "%s/%s=%d(%s) ",
                                 ladder_name[ai], pkn, (int)e.status, st_short((int)e.status));
                        combos += buf;
                        last_status = (int)e.status; last_msg = e.what();
                    }
                }
            }
            if (!have_pd) {
                // 全部算法族都失败：一行紧凑探针给出每个 alg/pk 的数值状态码，
                // 直接看出是哪个实现族坏（auto/direct 同时 oom ⇒ brgconv 嫌疑；
                // winograd=uni ⇒ 无 ACL 属预期）。首个形状额外打一次完整 hex。
                fprintf(stderr, "skip %d,%d,%d,%d,%d [PD-ALL] layout=%s: %s\n",
                        s.mb, s.ic, s.ih, s.iw, s.oc, layout, combos.c_str());
                if (first_fail_detail) {
                    dump_err(stderr, "PD-FIRST", s, last_status, last_msg);
                    first_fail_detail = false;
                }
                return false;
            }
            const char* impl = pd.impl_info_str();   // 实际选中的实现名（关键诊断）
            fprintf(stderr, "# %d,%d,%d,%d,%d via alg=%s pk=%s impl=%s layout=%s\n",
                    s.mb, s.ic, s.ih, s.iw, s.oc, via_alg, via_pk, impl, layout);

            dnnl::memory src_mem, wei_mem, bia_mem, dst_mem;
            try {
                src_mem = dnnl::memory(pd.src_desc(), eng);
                wei_mem = dnnl::memory(pd.weights_desc(), eng);
                bia_mem = dnnl::memory(pd.bias_desc(), eng);
                dst_mem = dnnl::memory(pd.dst_desc(), eng);
            } catch (const dnnl::error& e) { dump_err(stderr, "MEM", s, (int)e.status, e.what()); return false; }
            fill((float*)src_mem.get_data_handle(), src_mem.get_desc().get_size() / 4);
            fill((float*)wei_mem.get_data_handle(), wei_mem.get_desc().get_size() / 4);
            fill((float*)bia_mem.get_data_handle(), bia_mem.get_desc().get_size() / 4);

            dnnl::convolution_forward conv;
            try { conv = dnnl::convolution_forward(pd); }
            catch (const dnnl::error& e) { dump_err(stderr, "PRIM", s, (int)e.status, e.what()); return false; }

            // execute 的签名就是 std::unordered_map<int, memory>；oneDNN 没有
            // dnnl::execution_args 这个别名，直接写裸类型更稳。
            std::unordered_map<int, dnnl::memory> args{
                { DNNL_ARG_SRC, src_mem }, { DNNL_ARG_WEIGHTS, wei_mem },
                { DNNL_ARG_BIAS, bia_mem }, { DNNL_ARG_DST, dst_mem } };
            auto run_once = [&]() { conv.execute(stream, args); stream.wait(); };

            try { for (int i = 0; i < warmup; i++) run_once(); }
            catch (const dnnl::error& e) { dump_err(stderr, "EXEC_WARMUP", s, (int)e.status, e.what()); return false; }

            double best = 1e30;
            try {
                for (int i = 0; i < repeats; i++) {
                    auto t0 = std::chrono::steady_clock::now();
                    run_once();
                    double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                    best = std::min(best, ms);
                }
            } catch (const dnnl::error& e) {
                dump_err(stderr, "EXEC_REPEAT", s, (int)e.status, e.what()); return false;
            }
            fprintf(stdout, "%d,%d,%d,%d,%d,%.3f\n",
                    s.mb, s.ic, s.ih, s.iw, s.oc, best);
            return true;
        };

        // 首选 any 布局（端到端口径）；任何阶段失败 → 整条管线用显式 nchw/oihw
        // 重试保证出数据，并留 NOTE 说明该形状走的是兜底路径。
        if (!run_one("any")) {
            if (run_one("nchw"))
                fprintf(stderr, "NOTE %d,%d,%d,%d,%d: any-layout pipeline failed; explicit nchw/oihw fallback ok\n",
                        s.mb, s.ic, s.ih, s.iw, s.oc);
        }
    }
    return 0;
}
