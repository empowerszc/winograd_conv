// oneDNN 端到端 fp32 3x3 conv 基准 —— 与 bench_winograd 同 shape(CSV) 对照。
//
// 公平口径（docs/why_faster_than_acl_23.11.md §10.4）：
//   - primitive 跨迭代复用（不重建、权重变换由 oneDNN 内部缓存）
//   - src/dst/weights 全用 format_any，让 oneDNN 自选布局与算法
//   - 线程绑定交给外层脚本（OMP_PROC_BIND=close OMP_PLACES=cores）
// 计时语义与 compare.sh 对齐：warmup 后 repeats 次取 MIN（best-of-repeats）。
//
// 构建（脚本里自动做，等价于）：
//   $CXX -O3 -std=c++17 -fopenmp -I$ONEDNN_ROOT/include onednn_e2e.cpp \
//        -L$ONEDNN_ROOT/lib -Wl,-rpath,$ONEDNN_ROOT/lib -ldnnl -o build/onednn_e2e
// 用法：
//   onednn_e2e <shapes.csv> <threads> [warmup] [repeats] [--auto|--winograd]   CSV 模式
//   onednn_e2e --shape "mb,ic,ih,iw,oc" <threads> [warmup] [repeats] [--auto|--winograd]
//   onednn_e2e --diag <case> <threads>
// 输出：
//   # run: threads=16 warmup=3 repeats=20 alg=auto
//   mb,ic,ih,iw,oc,onednn_ms
//   （stdout 仅数据行；诊断/进度走 stderr）
//
// 2026-08-29 PD 构造修复：conv PD 一律用「带 bias、不带 dilates」的重载
// （dnnl.hpp 5740+ 第一个公开 ctor）。此前显式传 dilates={1,1}（CSV 是 dh=dw=0，
// 应传 {0,0}），等于给每个无膨胀 conv 加了 dilation=1；在集群
// onednn-3.12.1-release 这种不做 dst 校验的 build 里，错误 dilation 直达 impl 并
// 演变成 PD 创建 out_of_memory（[PD-ALL] 全 oom）。不带 dilates → 内部置 0。
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

// 解析 "mb,ic,ih,iw,oc"（--shape 单形状模式；kh/kw 等其余字段按 3x3 s1 p1 填充）
bool parse_shape_str(const char* s, Shape& out) {
    int n = sscanf(s, "%d,%d,%d,%d,%d",
                   &out.mb, &out.ic, &out.ih, &out.iw, &out.oc);
    if (n != 5) return false;
    out.kh = out.kw = 3;
    out.sh = out.sw = 1;
    out.ph = out.pw = 1;
    out.dh = out.dw = 0;
    out.grp = 1;
    return true;
}

// conv PD 构造 —— 采用用户参考程序的格式（2026-08-29 修复）：
//   dnnl::convolution_forward::primitive_desc(eng, pk, alg, src, wei, bias, dst,
//       strides, padding_l, padding_r)          // ← 无 dilates！
// 不带 dilates 的重载（dnnl.hpp 5740+ 的第一个公开 ctor）内部把 dilates 传 C API
// 时为 nullptr，conv_desc_init 会把它置 0（无膨胀）。此前本程序显式传
// dilate{1,1}，等于给每个无膨胀 conv 加了 dilation=1 —— 在集群
// onednn-3.12.1-release 这种不做 dst 校验的 build 里，错误 dilation 直达 impl，
// 算出异常大的分配 → PD 创建返回 out_of_memory（[PD-ALL] 全 oom 的根源）。
// bias 始终用具体格式 tag::x（一维 plain，最常规用法），不用 format_any。
// diag_dil1=true 仅用于 --diag 复现旧 bug（显式 dilates={1,1}）。
bool try_conv_pd(const Shape& s, dnnl::engine& eng, const char* layout,
        dnnl::algorithm alg, dnnl::prop_kind pk, bool diag_dil1,
        dnnl::convolution_forward::primitive_desc& pd, int& out_status,
        std::string& what) {
    using md = dnnl::memory::desc;
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const dnnl::memory::dims src_d{ s.mb, s.ic, s.ih, s.iw };
    const dnnl::memory::dims wei_d{ s.oc, s.ic, 3, 3 };
    const dnnl::memory::dims bia_d{ s.oc };
    const dnnl::memory::dims dst_d{ s.mb, s.oc, s.ih, s.iw };
    md s_md, w_md, b_md, d_md;
    if (std::strcmp(layout, "nchw") == 0) {   // 显式布局：强制 ref/gemm 路径
        s_md = md(src_d, dt::f32, tag::nchw);
        w_md = md(wei_d, dt::f32, tag::oihw);
        b_md = md(bia_d, dt::f32, tag::x);
        d_md = md(dst_d, dt::f32, tag::nchw);
    } else {                                  // format_tag::any：oneDNN 自选
        s_md = md(src_d, dt::f32, tag::any);
        w_md = md(wei_d, dt::f32, tag::any);
        b_md = md(bia_d, dt::f32, tag::x);
        d_md = md(dst_d, dt::f32, tag::any);
    }
    try {
        if (diag_dil1) {                      // --diag 复现旧 bug
            const dnnl::memory::dims stride{ 1, 1 }, dilate{ 1, 1 },
                    padl{ 1, 1 }, padr{ 1, 1 };
            pd = dnnl::convolution_forward::primitive_desc(eng, pk, alg,
                    s_md, w_md, b_md, d_md, stride, dilate, padl, padr);
        } else {                              // 用户参考格式：无 dilates
            const dnnl::memory::dims stride{ 1, 1 }, padl{ 1, 1 }, padr{ 1, 1 };
            pd = dnnl::convolution_forward::primitive_desc(eng, pk, alg,
                    s_md, w_md, b_md, d_md, stride, padl, padr);
        }
        return true;
    } catch (const dnnl::error& e) {
        out_status = (int)e.status;
        what = e.what();
        return false;
    }
}

// 探针（主基准之后运行，避免任何 PD 创建先于数据流；全部打在 stderr 供脚本 grep）。
void run_probes() {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const dnnl::memory::dims d{ 1, 16, 16, 16 };
    auto nchw_md = dnnl::memory::desc(d, dt::f32, tag::nchw);
    auto any_md  = dnnl::memory::desc(d, dt::f32, tag::any);
    dnnl::engine e0(dnnl::engine::kind::cpu, 0);
    // [preOMP] 第一个 eltwise PD（nchw）——正常应成功
    {
        try {
            auto epd = dnnl::eltwise_forward::primitive_desc(e0,
                    dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu,
                    nchw_md, nchw_md, 0.0f, 0.0f);
            fprintf(stderr, "[preOMP] eltwise nchw PD ok impl=%s\n",
                    epd.impl_info_str());
        } catch (const dnnl::error &e) {
            fprintf(stderr, "[preOMP] eltwise nchw PD FAIL status=%d(%s)\n",
                    (int)e.status, status_name((int)e.status));
        }
    }
    // [smoke] 第二个 eltwise PD（同样 nchw）——证明「第 1 个 PD 之后进程没被毒化」
    {
        try {
            auto epd = dnnl::eltwise_forward::primitive_desc(e0,
                    dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu,
                    nchw_md, nchw_md, 0.0f, 0.0f);
            fprintf(stderr, "[smoke] eltwise nchw (2nd PD) ok impl=%s\n",
                    epd.impl_info_str());
        } catch (const dnnl::error &e) {
            fprintf(stderr, "[smoke] eltwise nchw (2nd PD) FAIL status=%d(%s)\n",
                    (int)e.status, status_name((int)e.status));
        }
    }
    // [smoke-any] eltwise format_any —— 该 build 大概率不支持（非生产路径，仅记录；
    // 它 oom 是「格式 any 的 eltwise 不可用」的佐证，不是 conv oom 的根因）
    {
        try {
            auto epd = dnnl::eltwise_forward::primitive_desc(e0,
                    dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu,
                    any_md, any_md, 0.0f, 0.0f);
            fprintf(stderr, "[smoke-any] eltwise any PD ok impl=%s\n",
                    epd.impl_info_str());
        } catch (const dnnl::error &e) {
            fprintf(stderr,
                    "[smoke-any] eltwise any PD FAIL status=%d(%s) "
                    "(eltwise+format_any 该 build 不可用，预期，非数据路径)\n",
                    (int)e.status, status_name((int)e.status));
        }
    }
    // 库级最小分配 + 版本 + 线程探针
    {
        const auto *ver = dnnl::version();
        fprintf(stderr,
                "[env] header DNNL_VERSION=%d.%d.%d lib runtime=%d.%d.%d "
                "cpu_runtime=%u gpu_runtime=%u hash=%s\n",
                DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
                ver->major, ver->minor, ver->patch, ver->cpu_runtime,
                ver->gpu_runtime, ver->hash ? ver->hash : "(null)");
    }
#ifdef _OPENMP
    fprintf(stderr, "[thr] omp_get_max_threads=%d omp_get_num_procs=%d\n",
            omp_get_max_threads(), omp_get_num_procs());
#endif
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

// 单形状完整跑一遍（可指定布局）。任一步失败 → dump【阶段 + 数值状态码 +
// hex 字节】并返回 false。成功打印一行数据并返回 true。
bool run_one(const Shape& s, const char* layout, int warmup, int repeats,
        const std::string& alg_s) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(eng);

    std::vector<std::pair<dnnl::algorithm, const char*>> ladder;
    if (alg_s == "winograd")
        ladder = { { dnnl::algorithm::convolution_winograd, "winograd" },
                   { dnnl::algorithm::convolution_direct,   "direct" },
                   { dnnl::algorithm::convolution_auto,     "auto" } };
    else
        ladder = { { dnnl::algorithm::convolution_direct,   "direct" },
                   { dnnl::algorithm::convolution_auto,     "auto" },
                   { dnnl::algorithm::convolution_winograd, "winograd" } };

    dnnl::convolution_forward::primitive_desc pd;
    bool have_pd = false;
    const char* via_alg = "";
    const char* via_pk = "";
    std::string combos;         // 失败组合摘要 "alg/pk=st_short "（编码无关探针）
    int last_status = 0;
    std::string last_msg;
    for (const auto& la : ladder) {
        for (auto pk : { dnnl::prop_kind::forward_inference,
                         dnnl::prop_kind::forward }) {
            const char* pkn = (pk == dnnl::prop_kind::forward) ? "fwd" : "fwdinf";
            int st = 0;
            std::string what;
            if (try_conv_pd(s, eng, layout, la.first, pk, false, pd, st, what)) {
                have_pd = true; via_alg = la.second; via_pk = pkn;
                break;
            }
            char buf[96];
            snprintf(buf, sizeof buf, "%s/%s=%d(%s) ",
                     la.second, pkn, st, st_short(st));
            combos += buf;
            last_status = st; last_msg = what;
        }
        if (have_pd) break;
    }
    if (!have_pd) {
        fprintf(stderr, "skip %d,%d,%d,%d,%d [PD-ALL] layout=%s: %s\n",
                s.mb, s.ic, s.ih, s.iw, s.oc, layout, combos.c_str());
        dump_err(stderr, "PD-FIRST", s, last_status, last_msg);
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
}

// 单形状：首选 any 布局；任何阶段失败 → 整条管线用显式 nchw/oihw 重试保证出数据。
bool run_one_shape(const Shape& s, int warmup, int repeats, const std::string& alg_s) {
    if (!run_one(s, "any", warmup, repeats, alg_s)) {
        if (run_one(s, "nchw", warmup, repeats, alg_s)) {
            fprintf(stderr, "NOTE %d,%d,%d,%d,%d: any-layout pipeline failed; "
                    "explicit nchw/oihw fallback ok\n",
                    s.mb, s.ic, s.ih, s.iw, s.oc);
            return true;
        }
        return false;
    }
    return true;
}

// --diag <case>：单 PD 创建诊断（脚本按 case 分别起独立进程，避免跨 PD 状态干扰）。
//  cases: eltwise_nchw / eltwise_any / conv_direct_nchw_nodil /
//         conv_direct_any_nodil / conv_direct_any_dil1
int run_diag(const std::string& case_name) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    dnnl::engine e0(dnnl::engine::kind::cpu, 0);
    const dnnl::memory::dims d{ 1, 16, 16, 16 };
    if (case_name == "eltwise_nchw" || case_name == "eltwise_any") {
        auto smd = dnnl::memory::desc(d, dt::f32,
                case_name == "eltwise_any" ? tag::any : tag::nchw);
        try {
            auto epd = dnnl::eltwise_forward::primitive_desc(e0,
                    dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu,
                    smd, smd, 0.0f, 0.0f);
            fprintf(stdout, "[diag] %s ok impl=%s\n",
                    case_name.c_str(), epd.impl_info_str());
            return 0;
        } catch (const dnnl::error& e) {
            fprintf(stdout, "[diag] %s FAIL status=%d(%s)\n", case_name.c_str(),
                    (int)e.status, status_name((int)e.status));
            return (int)e.status;
        }
    }
    // conv cases（统一 direct + forward_inference = 用户参考组合，只变 布局/dilates）
    Shape s{ 1, 16, 16, 16, 16, 3, 3, 1, 1, 1, 1, 0, 0, 1 };
    const char* layout = (case_name == "conv_direct_nchw_nodil") ? "nchw" : "any";
    const bool dil1 = (case_name == "conv_direct_any_dil1");
    dnnl::convolution_forward::primitive_desc pd;
    int st = 0;
    std::string what;
    if (try_conv_pd(s, e0, layout, dnnl::algorithm::convolution_direct,
            dnnl::prop_kind::forward_inference, dil1, pd, st, what)) {
        fprintf(stdout, "[diag] %s ok impl=%s\n", case_name.c_str(), pd.impl_info_str());
        return 0;
    }
    fprintf(stdout, "[diag] %s FAIL status=%d(%s) what=%s\n", case_name.c_str(),
            st, status_name(st), what.c_str());
    return st;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  %s <shapes.csv> <threads> [warmup] [repeats] [--auto|--winograd]\n"
            "  %s --shape \"mb,ic,ih,iw,oc\" <threads> [warmup] [repeats] [--auto|--winograd]\n"
            "  %s --diag <case> <threads>\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    std::string mode = "csv";
    const char* csv_path = nullptr;
    std::string shape_override, diag_case;
    int argi = 1;
    if (std::string(argv[1]) == "--shape") {
        mode = "shape"; shape_override = argv[2]; argi = 3;
    } else if (std::string(argv[1]) == "--diag") {
        mode = "diag"; diag_case = argv[2]; argi = 3;
    } else {
        csv_path = argv[1]; argi = 2;
    }
    if (argi >= argc) { fprintf(stderr, "error: <threads> missing\n"); return 1; }
    int threads = atoi(argv[argi++]);
    int warmup = 3, repeats = 20;
    auto next_int = [&](int def) {
        if (argi < argc && argv[argi][0] >= '0' && argv[argi][0] <= '9')
            return atoi(argv[argi++]);
        return def;
    };
    warmup = next_int(3);
    repeats = next_int(20);
    std::string alg_s = "auto";
    for (; argi < argc; argi++) {
        if (std::string(argv[argi]) == "--winograd") alg_s = "winograd";
        else if (std::string(argv[argi]) == "--auto") alg_s = "auto";
    }

    // verbose：仅当 ONEDNN_VERBOSE 未设置时打 level-1（探针用 env 传 all 时不覆盖）
    if (getenv("ONEDNN_VERBOSE") == nullptr) dnnl::set_verbose(1);

    if (mode == "diag") return run_diag(diag_case);

    if (mode == "shape") {
        Shape s;
        if (!parse_shape_str(shape_override.c_str(), s)) {
            fprintf(stderr, "error: bad --shape '%s' (need mb,ic,ih,iw,oc)\n",
                    shape_override.c_str());
            return 1;
        }
        fprintf(stderr, "parsed 1 shape (%d,%d,%d,%d,%d)\n",
                s.mb, s.ic, s.ih, s.iw, s.oc);
        fprintf(stdout, "mb,ic,ih,iw,oc,onednn_ms\n");
        run_one_shape(s, warmup, repeats, alg_s);
        return 0;
    }

    // CSV 模式：先跑全部形状（数据流优先，PD 无 prior 干扰），探针最后。
    std::vector<Shape> shapes;
    if (!parse_csv(csv_path, shapes)) return 1;
    fprintf(stderr, "parsed %zu shapes\n", shapes.size());   // 自查：0 => parse 失败
    fprintf(stdout, "# run: threads=%d warmup=%d repeats=%d alg=%s\n",
            threads, warmup, repeats, alg_s.c_str());
    fprintf(stdout, "mb,ic,ih,iw,oc,onednn_ms\n");
    for (const auto& s : shapes)
        run_one_shape(s, warmup, repeats, alg_s);
    run_probes();
    return 0;
}
