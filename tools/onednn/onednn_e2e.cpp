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
#include <algorithm>
#include <chrono>
#include <cstdio>
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

    // 线程数交给 omp_set_num_threads + 外层 OMP_PROC_BIND/PLACES。
    // 不用 dnnl::set_max_threads：它随 DNNL_CPU_THREADING_RUNTIME 条件编译，
    // 部分发行包（如 3.12.1-release）里根本不存在。
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    dnnl::algorithm alg = (alg_s == "winograd")
        ? dnnl::algorithm::convolution_winograd
        : dnnl::algorithm::convolution_auto;

    fprintf(stdout, "# run: threads=%d warmup=%d repeats=%d alg=%s\n",
            threads, warmup, repeats, alg_s.c_str());
    fprintf(stdout, "mb,ic,ih,iw,oc,onednn_ms\n");

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
            int last_status = 0;
            std::string last_msg;
            for (auto pk : { dnnl::prop_kind::forward, dnnl::prop_kind::forward_inference }) {
                try {
                    pd = dnnl::convolution_forward::primitive_desc(
                        eng, pk, alg, s_md, w_md, b_md, d_md, stride, dilate, padl, padr);
                    have_pd = true;
                    break;
                } catch (const dnnl::error& e) {
                    last_status = (int)e.status; last_msg = e.what();
                }
            }
            if (!have_pd) { dump_err(stderr, "PD", s, last_status, last_msg); return false; }

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
