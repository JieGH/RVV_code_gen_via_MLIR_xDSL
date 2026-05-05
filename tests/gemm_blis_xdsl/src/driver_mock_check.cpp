#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

#include "gen/cache_config.h"

// Include the dispatcher which declares gemm_blis_B3A2C0
#include "gemm_macro_dispatch.h"

// --- Tile size constants matching the code generator (generate_macro.py) ---
// These are now sourced from the auto-generated gen/cache_config.h
static constexpr int MR = MR_OPT;
static constexpr int NR = NR_OPT;
static constexpr int MC = MC_OPT;
static constexpr int NC = NC_OPT;
static constexpr int KC = KC_OPT;

// --- Mock Implementations (Scalar counterparts for verification) ---

extern "C" {

// =========================================================================
// 1. Scalar Packing Implementations
// (Replaced by xDSL generated packing in Phase 4)
// =========================================================================

// pack_CB_v and pack_RB_v are now dynamically generated and linked from
// gemm_macro_v.c!

void gemm_microkernel_v(int mr, int nr, int kc, float alpha, float *Ac_ptr,
                        float *Bc_ptr, float betaI, float *Cptr, int ldC) {
  // IMPORTANT: The xDSL packing routines always pack into fixed-size blocks
  // of MR (for A) and NR (for B), regardless of the actual edge-tile size
  // mr/nr. We must use the FIXED packing strides MR and NR here.
  for (int j = 0; j < nr; ++j) {
    for (int i = 0; i < mr; ++i) {
      float ab = 0.0f;
      for (int p = 0; p < kc; ++p) {
        ab += Ac_ptr[p * MR + i] * Bc_ptr[p * NR + j];
      }
      // C(i,j) = alpha * A*B + beta * C
      Cptr[j * ldC + i] = alpha * ab + betaI * Cptr[j * ldC + i];
    }
  }
}

} // extern "C"

// --- Reference GEMM ---
void gemm_ref(int m, int n, int k, float alpha, float *A, int ldA, float *B,
              int ldB, float beta, float *C, int ldC) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float ab = 0.0f;
      for (int p = 0; p < k; ++p) {
        ab += A[p * ldA + i] * B[j * ldB + p];
      }
      C[j * ldC + i] = alpha * ab + beta * C[j * ldC + i];
    }
  }
}

int main() {
  const int M = 512;
  const int N = 512;
  const int K = 512;
  const float alpha = 1.0f;
  const float beta = 1.0f;

  // Col-major ldA=M, ldB=K, ldC=M
  std::vector<float> A(M * K), B(K * N), C_out(M * N), C_ref(M * N);

  // Initialize with deterministic values
  for (int i = 0; i < M * K; ++i)
    A[i] = (float)(i % 100) / 10.0f;
  for (int i = 0; i < K * N; ++i)
    B[i] = (float)(i % 70) / 7.0f;
  for (int i = 0; i < M * N; ++i) {
    float val = (float)(i % 50) / 5.0f;
    C_out[i] = val;
    C_ref[i] = val;
  }

  // Temporary pack buffers sized to MC*KC and NC*KC respectively
  std::vector<float> Ac(MC * KC), Bc(NC * KC);

  std::cout << "Starting xDSL Macro-Kernel Verification (M=" << M << ", N=" << N
            << ", K=" << K << ")..." << std::endl;

  // 1. Check Correctness First
  gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', M, N, K, alpha, A.data(), M,
                   B.data(), K, beta, C_out.data(), M, Ac.data(), Bc.data(),
                   MC, NC, KC, nullptr, nullptr, nullptr);

  gemm_ref(M, N, K, alpha, A.data(), M, B.data(), K, beta, C_ref.data(), M);

  float max_err = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    max_err = std::max(max_err, std::abs(C_out[i] - C_ref[i]));
  }

  if (max_err >= 2e-2) {
    std::cout << "❌ Verification FAILED! Max error: " << max_err << std::endl;
    return 1;
  }

  // -------------------------------------------------------------------------
  // 2. Performance Evaluation (Hot-Cache)
  // -------------------------------------------------------------------------
  const int NREP = 5;
  double min_time_xdsl = 1e9;
  double min_time_ref = 1e9;

  // Measure xDSL Scalar Fallback
  for (int r = 0; r < NREP; ++r) {
    auto start = std::chrono::high_resolution_clock::now();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', M, N, K, alpha, A.data(), M,
                     B.data(), K, beta, C_out.data(), M, Ac.data(), Bc.data(),
                     MC, NC, KC, nullptr, nullptr, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    min_time_xdsl = std::min(min_time_xdsl, duration.count());
  }

  // Measure Reference GEMM (Pure Scalar)
  for (int r = 0; r < NREP; ++r) {
    auto start = std::chrono::high_resolution_clock::now();
    gemm_ref(M, N, K, alpha, A.data(), M, B.data(), K, beta, C_ref.data(), M);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    min_time_ref = std::min(min_time_ref, duration.count());
  }

  double gflops_xdsl = (2.0 * M * N * K) / (min_time_xdsl * 1e6);
  double gflops_ref = (2.0 * M * N * K) / (min_time_ref * 1e6);

  std::cout << "✅ Verification SUCCESS! Max error: " << max_err << std::endl;
  std::cout << "⏱️  xDSL Scalar Perf: [ MIN: " << std::fixed << std::setprecision(2) << min_time_xdsl << " ms | " << gflops_xdsl << " GFLOPS ]" << std::endl;
  std::cout << "⏱️  Ref Scalar Perf:  [ MIN: " << std::fixed << std::setprecision(2) << min_time_ref << " ms | " << gflops_ref << " GFLOPS ]\n" << std::endl;

  return 0;
}
