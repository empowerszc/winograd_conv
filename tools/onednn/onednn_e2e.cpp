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
        try {
            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream stream(eng);

            using md = dnnl::memory::desc;
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;
            dnnl::memory::dims src_d{ s.mb, s.ic, s.ih, s.iw };
            dnnl::memory::dims wei_d{ s.oc, s.ic, 3, 3 };
            dnnl::memory::dims bia_d{ s.oc };
            dnnl::memory::dims dst_d{ s.mb, s.oc, s.ih, s.iw };
            md src_md(src_d, dt::f32, tag::any);
            md wei_md(wei_d, dt::f32, tag::any);
            md bia_md(bia_d, dt::f32, tag::any);
            md dst_md(dst_d, dt::f32, tag::any);
            dnnl::memory::dims stride{ 1, 1 }, dilate{ 1, 1 }, padl{ 1, 1 }, padr{ 1, 1 };

            // 实测 3.12.1-release（AArch64）对 forward_inference + any 的 conv pd
            // 创建全部失败，而 forward（training，benchdnn 同款路径）可用。依次尝试。
            dnnl::convolution_forward::primitive_desc pd;
            bool have_pd = false;
            std::string pd_err;
            for (auto pk : { dnnl::prop_kind::forward, dnnl::prop_kind::forward_inference }) {
                try {
                    pd = dnnl::convolution_forward::primitive_desc(
                        eng, pk, alg, src_md, wei_md, bia_md, dst_md,
                        stride, dilate, padl, padr);
                    have_pd = true;
                    break;
                } catch (const dnnl::error& e) {
                    pd_err = e.what();
                }
            }
            if (!have_pd)
                throw dnnl::error((dnnl_status_t)0, ("no conv pd: " + pd_err).c_str());

            dnnl::memory src_mem(pd.src_desc(), eng);
            dnnl::memory wei_mem(pd.weights_desc(), eng);
            dnnl::memory bia_mem(pd.bias_desc(), eng);
            dnnl::memory dst_mem(pd.dst_desc(), eng);
            fill((float*)src_mem.get_data_handle(), src_mem.get_desc().get_size() / 4);
            fill((float*)wei_mem.get_data_handle(), wei_mem.get_desc().get_size() / 4);
            fill((float*)bia_mem.get_data_handle(), bia_mem.get_desc().get_size() / 4);

            dnnl::convolution_forward conv(pd);
            // execute 的签名就是 std::unordered_map<int, memory>；oneDNN 没有
            // dnnl::execution_args 这个别名，直接写裸类型更稳。
            std::unordered_map<int, dnnl::memory> args{
                { DNNL_ARG_SRC, src_mem }, { DNNL_ARG_WEIGHTS, wei_mem },
                { DNNL_ARG_BIAS, bia_mem }, { DNNL_ARG_DST, dst_mem } };

            auto run_once = [&]() {
                conv.execute(stream, args);
                stream.wait();
            };
            for (int i = 0; i < warmup; i++) run_once();

            double best = 1e30;
            for (int i = 0; i < repeats; i++) {
                auto t0 = std::chrono::steady_clock::now();
                run_once();
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                best = std::min(best, ms);
            }
            fprintf(stdout, "%d,%d,%d,%d,%d,%.3f\n",
                    s.mb, s.ic, s.ih, s.iw, s.oc, best);
        } catch (const dnnl::error& e) {
            fprintf(stderr, "skip %d,%d,%d,%d,%d: dnnl error: %s\n",
                    s.mb, s.ic, s.ih, s.iw, s.oc, e.what());
        }
    }
    return 0;
}
