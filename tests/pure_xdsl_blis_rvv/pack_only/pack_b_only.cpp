// Issue #55: pack B and nothing else, rep after rep, to see whether a packer alone goes two-speed (#54).
//
//   pack_b_only M N K MR NR variant reps cpu
//
// variant: row-intr col-g1 col-g2 col-g4 col-gall col-g1-m4 (templates/pack_b_variants.h) or scalar (the canonical
// scalar packer, from scalar_pack.inc). `row` (the xDSL packer) is not linked here. cpu: a core number, or -1 for
// no pinning (host tests).
//
// One rep packs every (jc, pc) block of B exactly as the generated GEMM loop nest does, with the blocking the driver
// gets from get_optim_mc_nc_kc (no nc override): b_ptr = B + jc*ldb + pc, pack(b_ptr, Bc, cur_kc, cur_nc, ldb, NR).
// B is column-major, ldb = K, distinct values. Bc is allocated as driver_benchmark.cpp does. Five warm-up reps run
// first and are not in the output. One steady_clock read before and one after each rep; nothing else in the window.
// After the last rep Bc is compared with a replay of the whole loop through the scalar packer; exit 1 on a mismatch.
//
// Output (stdout): a `# key: value` header block, then `Rep,Pack_Seconds`, one row per rep, then `# check: ok`.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#include "pack_b_variants.h"
#include "scalar_pack.inc"

extern "C" {
#include "../modelLevel/model_level.h"
}

static const int WARMUP_REPS = 5;

static void* aligned_zeroed(size_t floats) {
    size_t bytes = (floats * sizeof(float) + 63) & ~(size_t)63;
    void* p = aligned_alloc(64, bytes);
    if (!p) { fprintf(stderr, "allocation failed\n"); exit(1); }
    memset(p, 0, bytes);
    return p;
}

struct Block { int32_t jc, pc, cur_nc, cur_kc; };

static void usage() {
    fprintf(stderr, "usage: pack_b_only M N K MR NR variant reps cpu\n"
                    "  variant: scalar row-intr col-g1 col-g2 col-g4 col-gall col-g1-m4\n");
}

int main(int argc, char** argv) {
    if (argc != 9) { usage(); return 2; }
    const int32_t M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    const int32_t MR = atoi(argv[4]), NR = atoi(argv[5]);
    const std::string vname = argv[6];
    const long reps = atol(argv[7]);
    const int cpu = atoi(argv[8]);
    if (M < 1 || N < 1 || K < 1 || MR < 1 || NR < 1 || reps < 1) { usage(); return 2; }

    int variant = -1;  // -1 = scalar
    bool scalar = (vname == "scalar");
    if (!scalar) {
        variant = pack_b_variant_from_name(vname.c_str());
        if (variant == PACK_B_ROW) {
            fprintf(stderr, "variant 'row' is the xDSL packer and is not linked into the pack-only loop; use row-intr\n");
            return 2;
        }
        if (variant < 0) { fprintf(stderr, "unknown variant '%s'\n", vname.c_str()); usage(); return 2; }
    }

    if (cpu >= 0) {
#ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) { perror("sched_setaffinity"); return 2; }
#else
        fprintf(stderr, "cannot pin on this platform\n");
        return 2;
#endif
    }

    int mc, nc, kc;
    get_optim_mc_nc_kc((int)sizeof(float), M, N, K, MR, NR, &mc, &nc, &kc);

    const int32_t ldb = K;
    std::vector<float> B((size_t)N * ldb);
    for (size_t i = 0; i < B.size(); ++i) B[i] = 1.0f + (float)i;  // distinct (exact up to 2^24 values)

    const size_t padded_nc = ((size_t)nc + NR - 1) / NR * NR;
    const size_t bc_floats = std::max<size_t>((size_t)(1 + kc) * (NR + nc), padded_nc * kc);
    float* Bc = (float*)aligned_zeroed(bc_floats);
    float* Bc_ref = (float*)aligned_zeroed(bc_floats);

    std::vector<Block> blocks;
    for (int32_t jc = 0; jc < N; jc += nc)
        for (int32_t pc = 0; pc < K; pc += kc)
            blocks.push_back({jc, pc, std::min<int32_t>(N - jc, nc), std::min<int32_t>(K - pc, kc)});

    auto one_rep = [&](float* dst) {
        for (const Block& b : blocks) {
            float* b_ptr = B.data() + (size_t)b.jc * ldb + b.pc;
            if (scalar) pack_CB_scalar(b.cur_kc, b.cur_nc, b_ptr, ldb, dst, NR);
            else pack_CB_variant(variant, b_ptr, dst, b.cur_kc, b.cur_nc, ldb, NR);
        }
    };

    for (int w = 0; w < WARMUP_REPS; ++w) one_rep(Bc);

    std::vector<double> secs((size_t)reps);
    for (long r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        one_rep(Bc);
        asm volatile("" ::: "memory");
        auto t1 = std::chrono::steady_clock::now();
        secs[(size_t)r] = std::chrono::duration<double>(t1 - t0).count();
    }

    // Check: replay the whole loop through the scalar packer into a fresh zeroed buffer, compare everything.
    for (const Block& b : blocks)
        pack_CB_scalar(b.cur_kc, b.cur_nc, B.data() + (size_t)b.jc * ldb + b.pc, ldb, Bc_ref, NR);
    const bool same = std::memcmp(Bc, Bc_ref, bc_floats * sizeof(float)) == 0;

    char host[256] = "unknown";
    gethostname(host, sizeof(host) - 1);
    printf("# board: %s\n", host);
    printf("# cpu: %d\n", cpu);
    printf("# variant: %s\n", vname.c_str());
    printf("# shape: M=%d N=%d K=%d\n", M, N, K);
    printf("# tile: MR=%d NR=%d\n", MR, NR);
    printf("# blocking: mc=%d nc=%d kc=%d\n", mc, nc, kc);
    printf("# reps: %ld (warm-up %d, not printed)\n", reps, WARMUP_REPS);
    printf("# blocks: %zu\n", blocks.size());
    for (const Block& b : blocks) printf("# block: jc=%d pc=%d nc=%d kc=%d\n", b.jc, b.pc, b.cur_nc, b.cur_kc);
    printf("Rep,Pack_Seconds\n");
    for (long r = 0; r < reps; ++r) printf("%ld,%.9g\n", r, secs[(size_t)r]);
    printf("# check: %s\n", same ? "ok" : "MISMATCH");
    if (!same) fprintf(stderr, "pack_b_only: Bc differs from the scalar packer's\n");
    return same ? 0 : 1;
}
