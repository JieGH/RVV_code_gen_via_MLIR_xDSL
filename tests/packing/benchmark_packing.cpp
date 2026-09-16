/**
 * benchmark_packing.cpp
 *
 * Benchmarks pack_CB_v and pack_RB_v against the scalar BLIS reference.
 * Measures Time, Equivalent GFLOPs, and Memory Bandwidth.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#ifdef TEST_INT8
typedef int8_t TestType;
#else
typedef float TestType;
#endif

#include "pack_dispatch.h"

// Scalar references
void ref_pack_CB(char orderM, char transM, int mc, int nc,
                 TestType* M, int ldM, TestType* Mc, int RR);
void ref_pack_RB(char orderM, char transM, int mc, int nc,
                 TestType* M, int ldM, TestType* Mc, int RR);

// The exact scalar references for INT8 Method 3
void ref_pack_DOT_INT8_LANE_Pack_CB(char orderA, char transM, int mc, int nc, TestType *M, int ldM, TestType *Mc, int nr_block, int KR);
void ref_pack_DOT_INT8_LANE_Pack_RB(char orderA, char transM, int mc, int nc, TestType *M, int ldM, TestType *Mc, int mr_block, int KR);

// NR and MR are injected at compile time via -DNR=<n> -DMR=<m> (see Makefile).
// compile_packing.sh is the single source of truth for block sizes.
// Defaults below are only a fallback for editors / static analysis.
#ifndef NR
#define NR 8
#endif
#ifndef MR
#define MR 8
#endif

#ifndef KR
#define KR 32
#endif

static int cb_packed_size(int mc, int nc) { 
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        return ((mc + KR - 1) / KR) * KR * ((nc + NR - 1) / NR) * NR;
    }
    return mc * ((nc + NR - 1) / NR) * NR; 
}
static int rb_packed_size(int mc, int nc) { 
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        return ((nc + KR - 1) / KR) * KR * ((mc + MR - 1) / MR) * MR;
    }
    return ((mc + MR - 1) / MR) * MR * nc; 
}

// Timer helper
double get_time(const std::chrono::high_resolution_clock::time_point& start,
                const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
}

void bench_CB(const char* label, char orderM, char transM, int mc, int nc, int ldM, int iters) {
    int psz = cb_packed_size(mc, nc);
    std::vector<TestType> B((size_t)ldM * std::max(mc, nc), 1);
    std::vector<TestType> vec(psz, 0);
    std::vector<TestType> ref(psz, 0);

    // Warmup
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        ref_pack_DOT_INT8_LANE_Pack_CB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), NR, KR);
    } else {
        ref_pack_CB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), NR);
    }
    pack_CB_v(orderM, transM, mc, nc, B.data(), ldM, vec.data(), NR);

    // Bench Ref
    int iters_ref = 0;
    auto start_ref = std::chrono::high_resolution_clock::now();
    while (true) {
        for(int i=0; i<10; ++i) {
            if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
                ref_pack_DOT_INT8_LANE_Pack_CB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), NR, KR);
            } else {
                ref_pack_CB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), NR);
            }
        }
        iters_ref += 10;
        auto current = std::chrono::high_resolution_clock::now();
        if (get_time(start_ref, current) >= 3.0) break;
    }
    auto end_ref = std::chrono::high_resolution_clock::now();
    double time_ref = get_time(start_ref, end_ref) / iters_ref;

    // Bench Vec
    int iters_vec = 0;
    auto start_vec = std::chrono::high_resolution_clock::now();
    while (true) {
        for(int i=0; i<10; ++i) {
            pack_CB_v(orderM, transM, mc, nc, B.data(), ldM, vec.data(), NR);
        }
        iters_vec += 10;
        auto current = std::chrono::high_resolution_clock::now();
        if (get_time(start_vec, current) >= 3.0) break;
    }
    auto end_vec = std::chrono::high_resolution_clock::now();
    double time_vec = get_time(start_vec, end_vec) / iters_vec;

    double bytes = (double)mc * nc * sizeof(TestType);
    double gflops_ref = (2.0 * mc * nc) / (time_ref * 1e9);
    double gflops_vec = (2.0 * mc * nc) / (time_vec * 1e9);
    double bw_ref = bytes / (time_ref * 1e9);
    double bw_vec = bytes / (time_vec * 1e9);

    std::printf("%-20s | %c,%c | %4d x %-4d | %8.2f | %8.2f | %9.3f | %9.3f | %8.2f | %8.2f | %5.2fx\n",
                label, orderM, transM, mc, nc, time_ref*1e6, time_vec*1e6, gflops_ref, gflops_vec, bw_ref, bw_vec, time_ref/time_vec);
}

void bench_RB(const char* label, char orderM, char transM, int mc, int nc, int ldM, int iters) {
    int psz = rb_packed_size(mc, nc);
    std::vector<TestType> B((size_t)ldM * std::max(mc, nc), 1);
    std::vector<TestType> vec(psz, 0);
    std::vector<TestType> ref(psz, 0);

    // Warmup
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        ref_pack_DOT_INT8_LANE_Pack_RB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), MR, KR);
    } else {
        ref_pack_RB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), MR);
    }
    pack_RB_v(orderM, transM, mc, nc, B.data(), ldM, vec.data(), MR);

    // Bench Ref
    int iters_ref = 0;
    auto start_ref = std::chrono::high_resolution_clock::now();
    while (true) {
        for(int i=0; i<10; ++i) {
            if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
                ref_pack_DOT_INT8_LANE_Pack_RB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), MR, KR);
            } else {
                ref_pack_RB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), MR);
            }
        }
        iters_ref += 10;
        auto current = std::chrono::high_resolution_clock::now();
        if (get_time(start_ref, current) >= 3.0) break;
    }
    auto end_ref = std::chrono::high_resolution_clock::now();
    double time_ref = get_time(start_ref, end_ref) / iters_ref;

    // Bench Vec
    int iters_vec = 0;
    auto start_vec = std::chrono::high_resolution_clock::now();
    while (true) {
        for(int i=0; i<10; ++i) {
            pack_RB_v(orderM, transM, mc, nc, B.data(), ldM, vec.data(), MR);
        }
        iters_vec += 10;
        auto current = std::chrono::high_resolution_clock::now();
        if (get_time(start_vec, current) >= 3.0) break;
    }
    auto end_vec = std::chrono::high_resolution_clock::now();
    double time_vec = get_time(start_vec, end_vec) / iters_vec;

    double bytes = (double)mc * nc * sizeof(TestType);
    double gflops_ref = (2.0 * mc * nc) / (time_ref * 1e9);
    double gflops_vec = (2.0 * mc * nc) / (time_vec * 1e9);
    double bw_ref = bytes / (time_ref * 1e9);
    double bw_vec = bytes / (time_vec * 1e9);

    std::printf("%-20s | %c,%c | %4d x %-4d | %8.2f | %8.2f | %9.3f | %9.3f | %8.2f | %8.2f | %5.2fx\n",
                label, orderM, transM, mc, nc, time_ref*1e6, time_vec*1e6, gflops_ref, gflops_vec, bw_ref, bw_vec, time_ref/time_vec);
}

int main() {
    std::printf("========================================================================================================================\n");
    std::printf("  Packing Benchmarks: Scalar vs Vector (NR=%d, MR=%d, Method=%s)\n", NR, MR, PACKING_METHOD);
    std::printf("========================================================================================================================\n");
    std::printf("%-20s | %-3s | %-11s | %-8s | %-8s | %-9s | %-9s | %-8s | %-8s | %-6s\n",
                "Kernel", "Ord", "Dimensions", "TimeRef", "TimeVec", "GFLOPsRef", "GFLOPsVec", "BWRef(GB)", "BWVec(GB)", "Speedup");
    std::printf("                     |     |             |   (us)   |   (us)   |           |           |           |           |        \n");
    std::printf("------------------------------------------------------------------------------------------------------------------------\n");

    int iters = 0; // Not used anymore, dynamically calculated based on 3 seconds
    int ldM = 4096; // Large ldM to simulate realistic GEMM memory spacing and fit large tests
    
    // CB if-branch (Col-Major) - col block
    bench_CB("CB if-branch (Col)", 'C', 'N', 256, 8, ldM, iters);
    bench_CB("CB if-branch (Col)", 'C', 'N', 1024, 8, ldM, iters);
    bench_CB("CB if-branch (Col)", 'C', 'N', 2048, 8, ldM, iters); // Rectangular 2K
    bench_CB("CB if-branch (Col)", 'C', 'N', 3072, 8, ldM, iters); // Rectangular 3K
    bench_CB("CB if-branch (Col)", 'C', 'N', 2048, 2048, ldM, iters); // Square 2K
    bench_CB("CB if-branch (Col)", 'C', 'N', 3072, 3072, ldM, iters); // Square 3K
    
    // CB else-branch (Row-Major) - row block
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") != 0) {
        bench_CB("CB else-branch (Row)", 'R', 'N', 256, 8, ldM, iters);
        bench_CB("CB else-branch (Row)", 'R', 'N', 1024, 8, ldM, iters);
        bench_CB("CB else-branch (Row)", 'R', 'N', 2048, 8, ldM, iters); // Rectangular 2K
        bench_CB("CB else-branch (Row)", 'R', 'N', 3072, 8, ldM, iters); // Rectangular 3K
        bench_CB("CB else-branch (Row)", 'R', 'N', 2048, 2048, ldM, iters); // Square 2K
        bench_CB("CB else-branch (Row)", 'R', 'N', 3072, 3072, ldM, iters); // Square 3K
    }

    std::printf("------------------------------------------------------------------------------------------------------------------------\n");

    // RB if-branch (Col-Major) - col block
    bench_RB("RB if-branch (Col)", 'C', 'N', 8, 256, ldM, iters);
    bench_RB("RB if-branch (Col)", 'C', 'N', 8, 1024, ldM, iters);
    bench_RB("RB if-branch (Col)", 'C', 'N', 8, 2048, ldM, iters); // Rectangular 2K
    bench_RB("RB if-branch (Col)", 'C', 'N', 8, 3072, ldM, iters); // Rectangular 3K
    bench_RB("RB if-branch (Col)", 'C', 'N', 2048, 2048, ldM, iters); // Square 2K
    bench_RB("RB if-branch (Col)", 'C', 'N', 3072, 3072, ldM, iters); // Square 3K

    // RB else-branch (Row-Major) - row block
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") != 0) {
        bench_RB("RB else-branch (Row)", 'R', 'N', 8, 256, ldM, iters);
        bench_RB("RB else-branch (Row)", 'R', 'N', 8, 1024, ldM, iters);
        bench_RB("RB else-branch (Row)", 'R', 'N', 8, 2048, ldM, iters); // Rectangular 2K
        bench_RB("RB else-branch (Row)", 'R', 'N', 8, 3072, ldM, iters); // Rectangular 3K
        bench_RB("RB else-branch (Row)", 'R', 'N', 2048, 2048, ldM, iters); // Square 2K
        bench_RB("RB else-branch (Row)", 'R', 'N', 3072, 3072, ldM, iters); // Square 3K
    }

    std::printf("========================================================================================================================\n");
    return 0;
}
