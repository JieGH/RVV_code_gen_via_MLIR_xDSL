#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "gen/cache_config.h"

// Dispatcher for the macro kernel
#include "gemm_macro_dispatch.h"

// Dispatch matrix from internal kernels directory
#include "../kernels/exo_matrix_RVV_fp32.h"

// --- Tile size constants matching the code generator (generate_macro.py) ---
// These are now sourced from the auto-generated gen/cache_config.h
static constexpr int MR = MR_OPT;
static constexpr int NR = NR_OPT;
static constexpr int MC = MC_OPT;
static constexpr int NC = NC_OPT;
static constexpr int KC = KC_OPT;

// Global dispatch table — initialized in main()
static ukrFunction ****ukr_matrix = nullptr;

extern "C" {

// =========================================================================
// 1. RVV Microkernel Wrapper
// Bridges the macro-kernel's call to the dynamically generated RVV microkernels
// =========================================================================

void gemm_microkernel_v(int mr, int nr, int kc, float alpha, float *Ac_ptr,
                        float *Bc_ptr, float betaI, float *Cptr, int ldC) {
  int beta_mode = (betaI == 0.0f) ? 0 : 1;
  ukrFunction fn = *ukr_matrix[mr][nr][beta_mode];

  if (fn != nullptr) {
    // Standard packed case: lda=MR, ldb=NR
    fn(nullptr, kc, &alpha, Ac_ptr, MR, Bc_ptr, NR, &betaI, Cptr, ldC);
  } else {
    // Fallback for edge cases without a compiled microkernel
    for (int j = 0; j < nr; ++j) {
      for (int i = 0; i < mr; ++i) {
        float ab = 0.0f;
        for (int p = 0; p < kc; ++p)
          ab += Ac_ptr[p * MR + i] * Bc_ptr[p * NR + j];
        Cptr[j * ldC + i] = alpha * ab + betaI * Cptr[j * ldC + i];
      }
    }
  }
}

// Specialized wrapper for unpacked testing with arbitrary strides
void gemm_microkernel_unpacked_v(int mr, int nr, int kc, float alpha,
                                 float *A_ptr, int ldA, float *B_ptr, int ldB,
                                 float betaI, float *Cptr, int ldC) {
  int beta_mode = (betaI == 0.0f) ? 0 : 1;
  ukrFunction fn = *ukr_matrix[mr][nr][beta_mode];

  if (fn != nullptr) {
    fn(nullptr, kc, &alpha, A_ptr, ldA, B_ptr, ldB, &betaI, Cptr, ldC);
  } else {
    for (int j = 0; j < nr; ++j) {
      for (int i = 0; i < mr; ++i) {
        float ab = 0.0f;
        for (int p = 0; p < kc; ++p)
          ab += A_ptr[p * ldA + i] * B_ptr[p * ldB + j];
        Cptr[j * ldC + i] = alpha * ab + betaI * Cptr[j * ldC + i];
      }
    }
  }
}

} // extern "C"

// =========================================================================
// 2. Unpacked RVV Test (No Blocking, No Packing)
// =========================================================================

void gemm_unpacked_rvv(int M, int N, int K, float alpha, float *A_col, int ldA,
                       float *B_row, int ldB, float beta, float *C_col,
                       int ldC) {
  for (int j = 0; j < N; j += NR) {
    int nr = std::min(NR, N - j);
    for (int i = 0; i < M; i += MR) {
      int mr = std::min(MR, M - i);

      // Pointers to the start of the current MRxK and KxNR micro-tiles
      float *A_ptr = &A_col[i];
      float *B_ptr = &B_row[j];
      float *C_ptr = &C_col[j * ldC + i];

      // Call the specialized unpacked wrapper
      gemm_microkernel_unpacked_v(mr, nr, K, alpha, A_ptr, ldA, B_ptr, ldB, beta,
                                  C_ptr, ldC);
    }
  }
}

// =========================================================================
// 3. Reference GEMM
// =========================================================================

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

// =========================================================================
// 3b. Hybrid Pipeline (Scalar Pack + RVV Ukernel)
// =========================================================================

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
static void ref_pack_CB(char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR) {
    int i, j, jj, k, nr;
    k = 0;
    if (((transM=='N')&&(orderM=='C')) || ((transM=='T')&&(orderM=='R'))) {
        for (j=0; j<nc; j+=RR) {
            k = j*mc;
            nr = std::min(nc-j, RR);
            for (i=0; i<mc; i++) {
                for (jj=0; jj<nr; jj++) { Mc[k++] = Mcol(i,j+jj); }
                k += (RR-nr);
            }
        }
    } else {
        for (j=0; j<nc; j+=RR) {
            k = j*mc;
            nr = std::min(nc-j, RR);
            for (i=0; i<mc; i++) {
                for (jj=0; jj<nr; jj++) { Mc[k++] = Mcol(j+jj,i); }
                k += (RR-nr);
            }
        }
    }
}

static void ref_pack_RB(char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR) {
    int i, j, ii, k, rr;
    if (((transM=='N')&&(orderM=='C')) || ((transM=='T')&&(orderM=='R'))) {
        for (i=0; i<mc; i+=RR) {
            k = i*nc;
            rr = std::min(mc-i, RR);
            for (j=0; j<nc; j++) {
                for (ii=0; ii<rr; ii++) { Mc[k++] = Mcol(i+ii, j); }
                k += (RR-rr);
            }
        }
    } else {
        for (i=0; i<mc; i+=RR) {
            k = i*nc;
            rr = std::min(mc-i, RR);
            for (j=0; j<nc; j++) {
                for (ii=0; ii<rr; ii++) { Mc[k++] = Mcol(j, i+ii); }
                k += (RR-rr);
            }
        }
    }
}
#undef Mcol

// Adapter matching gemm_ukr_ft signature that routes to the explicitly-typed gemm_microkernel_v
extern "C" void gemm_microkernel_v(int mr, int nr, int kc, float alpha, float *Ac_ptr, float *Bc_ptr, float betaI, float *Cptr, int ldC);
static void rvv_ukr_adapter(int mr, int nr, int kc, const void *alpha, const void *a, const void *b, const void *beta, void *c, int inc_c, int ld_c, const void *aux, const void *cntx) {
    gemm_microkernel_v(mr, nr, kc, *(const float*)alpha, (float*)a, (float*)b, *(const float*)beta, (float*)c, ld_c);
}

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]

static void ref_gemm_blis_B3A2C0(char orderA, char orderB, char orderC,
                                 char transA, char transB, size_t m, size_t n, size_t k, 
                                 float alpha, float *A, size_t ldA, float *B, size_t ldB, 
                                 float beta, float *C, size_t ldC, float *Ac, float *Bc, 
                                 size_t MC_, size_t NC_, size_t KC_, int MR_arg, int NR_arg,
                                 gemm_ukr_ft gemm_kernel) {
  size_t ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float zero = 0.0f, one = 1.0f, betaI; 
  float *Aptr, *Bptr, *Cptr;
   
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) ) return;
  
  for ( jc=0; jc<n; jc+=NC_ ) {
    nc = std::min(n-jc, NC_); 
    for ( pc=0; pc<k; pc+=KC_ ) {
      kc = std::min(k-pc, KC_); 
      
      if ( (transB=='N')&&(orderB=='C') )      Bptr = &Bcol(pc,jc);
      else if ( (transB=='N')&&(orderB=='R') ) Bptr = &Brow(pc,jc);
      else if ( (transB=='T')&&(orderB=='C') ) Bptr = &Bcol(jc,pc);
      else                                     Bptr = &Brow(jc,pc);
      
      ref_pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR_arg);
      
      if ( pc==0 ) betaI = beta;
      else         betaI = one;
      
      for ( ic=0; ic<m; ic+=MC_ ) {
        mc = std::min(m-ic, MC_); 
        
        if ( (transA=='N')&&(orderA=='C') )      Aptr = &Acol(ic, pc);
        else if ( (transA=='N')&&(orderA=='R') ) Aptr = &Arow(ic, pc);
        else if ( (transA=='T')&&(orderA=='C') ) Aptr = &Acol(pc, ic);
        else                                     Aptr = &Arow(pc, ic);
        
        ref_pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR_arg);
        
        for ( jr=0; jr<nc; jr+=NR_arg ) {
          nr = std::min(nc-jr, (size_t)NR_arg); 
          for ( ir=0; ir<mc; ir+=MR_arg ) {
            mr = std::min(mc-ir, (size_t)MR_arg); 
            
            if ( orderC=='C' ) Cptr = &Ccol(ic+ir,jc+jr);
            else               Cptr = &Crow(ic+ir,jc+jr);
            
            gemm_kernel(mr, nr, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, 1, ldC, nullptr, nullptr);
          }
        }
      }
    }
  }
}
#undef Acol
#undef Bcol
#undef Ccol
#undef Arow
#undef Brow
#undef Crow

// =========================================================================
// 4. Analytical Cache Model  (matches reference model_level.c / bananapi.h)
//    K1/BananaPi: NL1=128,CL1=64,WL1=4 | NL2=1024,CL2=64,WL2=8
//                 NL3=4096,CL3=64,WL3=16
//    Formula: Low et al. (2016) — subtract 1 from WL for non-resident operand
// =========================================================================

static int cache_model_level(bool isL3, int NL, int CL, int WL, int ds,
                              int m, int n) {
  int k, CAr, CBr;
  if (WL == 2) {
    k = isL3 ? NL*CL/(2*n*ds) : NL*CL/(2*m*ds);
  } else {
    CAr = (int)std::floor((WL - 1.0) / (1.0 + (double)n / m));
    if (CAr == 0) {
      CAr = 1; CBr = WL - 2;
      k = isL3 ? CBr*NL*CL/(m*ds) : CBr*NL*CL/(n*ds);
    } else {
      CBr = (int)std::ceil((double)n / m * CAr);
      k = isL3 ? CBr*NL*CL/(n*ds) : CAr*NL*CL/(m*ds);
    }
  }
  return k;
}

static void per_layer_mc_nc_kc(int M, int N, int K,
                                int *mc_opt, int *nc_opt, int *kc_opt) {
  // BananaPi/K1 cache parameters (matches archs/bananapi.h)
  const int NL1=128,CL1=64,WL1=4;
  const int NL2=1024,CL2=64,WL2=8;
  const int NL3=4096,CL3=64,WL3=16;
  const int ds = 4; // float = 4 bytes

  int kc = cache_model_level(false, NL1,CL1,WL1, ds, MR, NR);
  kc = std::min(K, kc);
  int mc = cache_model_level(false, NL2,CL2,WL2, ds, kc, NR);
  mc = std::min(M, mc);
  int nc = cache_model_level(true,  NL3,CL3,WL3, ds, kc, mc);
  nc = std::min(N, nc);

  *kc_opt = kc;  *mc_opt = mc;  *nc_opt = nc;
}

// =========================================================================
// 5. Test Runner
// =========================================================================

// Helper: 32-byte aligned float allocation (zero-initialized)
static float* aligned_float_alloc(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 32, count * sizeof(float)) != 0) {
        std::cerr << "aligned alloc failed\n"; std::exit(1);
    }
    std::memset(ptr, 0, count * sizeof(float));
    return static_cast<float*>(ptr);
}

bool run_test(int M, int N, int K, float alpha, float beta, const char *label,
              int force_mc = 0, int force_nc = 0, int force_kc = 0) {
  auto A = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * K), free);
  auto B = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(K * N), free);
  auto C_ref = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * N), free);
  auto C_packed = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * N), free);
  auto C_hybrid = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * N), free);
  auto C_unpacked = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * N), free);

  // Initialize matrices
  for (int i = 0; i < M * K; ++i) A.get()[i] = (float)(i % 100) / 10.0f;
  for (int i = 0; i < K * N; ++i) B.get()[i] = (float)(i % 70) / 7.0f;
  for (int i = 0; i < M * N; ++i) {
    float val = (float)(i % 50) / 5.0f;
    C_ref.get()[i] = C_packed.get()[i] = C_hybrid.get()[i] = C_unpacked.get()[i] = val;
  }

  // B_row: row-major transposed B for unpacked test
  auto B_row = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(K * N), free);
  for (int j = 0; j < N; ++j)
    for (int p = 0; p < K; ++p)
      B_row.get()[p * N + j] = B.get()[j * K + p];

  // Per-layer analytical optimal MC/NC/KC (reference methodology)
  int mc_calc, nc_calc, kc_calc;
  per_layer_mc_nc_kc(M, N, K, &mc_calc, &nc_calc, &kc_calc);

  // Default to analytical
  int mc_opt = mc_calc, nc_opt = nc_calc, kc_opt = kc_calc;
  bool is_overridden = false;

  // 1. Allow per-layer injection from .dat file parsing
  if (force_mc > 0) { mc_opt = force_mc; is_overridden = true; }
  if (force_nc > 0) { nc_opt = force_nc; is_overridden = true; }
  if (force_kc > 0) { kc_opt = force_kc; is_overridden = true; }

  // 2. Allow manual injection from bash environment (highest priority)
  if (const char* env_mc = std::getenv("OVERRIDE_MC")) { mc_opt = std::stoi(env_mc); is_overridden = true; }
  if (const char* env_nc = std::getenv("OVERRIDE_NC")) { nc_opt = std::stoi(env_nc); is_overridden = true; }
  if (const char* env_kc = std::getenv("OVERRIDE_KC")) { kc_opt = std::stoi(env_kc); is_overridden = true; }

  // Pack buffers — sized to compiled MC/KC and NC/KC, but padded with guard rows
  const int KR = MR;
  const size_t ac_size = std::max((size_t)(MC * KC),
                                  (size_t)(MR + mc_opt) * (KR + kc_opt));
  const size_t bc_size = std::max((size_t)(NC * KC),
                                  (size_t)(KR + kc_opt) * (NR + nc_opt));
  auto Ac = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(ac_size), free);
  auto Bc = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(bc_size), free);

  // =========================================================================
  // 1. Correctness check (one pass each)
  // =========================================================================
  gemm_ref(M, N, K, alpha, A.get(), M, B.get(), K, beta, C_ref.get(), M);

  gemm_blis_v_CN_CN_C_fp32(M, N, K, alpha, A.get(), M, B.get(), K, beta, C_packed.get(), M, Ac.get(), Bc.get());

  ref_gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', M, N, K, alpha, A.get(), M,
                       B.get(), K, beta, C_hybrid.get(), M, Ac.get(), Bc.get(),
                       mc_opt, nc_opt, kc_opt, MR, NR, rvv_ukr_adapter);

  gemm_unpacked_rvv(M, N, K, alpha, A.get(), M, B_row.get(), N, beta,
                   C_unpacked.get(), M);

  float err_packed = 0.0f, err_hybrid = 0.0f, err_unpacked = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    err_packed   = std::max(err_packed,   std::abs(C_packed.get()[i]   - C_ref.get()[i]));
    err_hybrid   = std::max(err_hybrid,   std::abs(C_hybrid.get()[i]   - C_ref.get()[i]));
    err_unpacked = std::max(err_unpacked, std::abs(C_unpacked.get()[i] - C_ref.get()[i]));
  }

  // Adaptive tolerance: FP32 accumulation error scales linearly with K.
  // Empirically: max_abs_err ≈ K × 4e-6 for matrix values in [0, 10].
  // Floor of 1e-3 handles very small K where rounding noise dominates.
  const float tol = std::max(1e-3f, (float)K * 5e-6f);
  const bool pass_p = (err_packed   < tol);
  const bool pass_h = (err_hybrid   < tol);
  const bool pass_u = (err_unpacked < tol);

  // =========================================================================
  // 2. Performance — adaptive average timing (matches reference methodology)
  //    Repeat until >= TMIN_SEC wall-clock seconds elapsed; report average.
  //    Never skipped, even when correctness check warns.
  // =========================================================================
  const double TMIN_SEC = 1.0;

  // -- Packed --
  int nrep_p = 0; double total_p_ns = 0.0;
  while (nrep_p < 3 || total_p_ns < TMIN_SEC * 1e9) {
    auto s = std::chrono::high_resolution_clock::now();
    gemm_blis_v_CN_CN_C_fp32(M, N, K, alpha, A.get(), M, B.get(), K, beta, C_packed.get(), M, Ac.get(), Bc.get());
    auto e = std::chrono::high_resolution_clock::now();
    total_p_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    ++nrep_p;
  }
  const double gflops_p = (2.0 * M * N * K) / (total_p_ns / nrep_p);

  // -- Hybrid --
  int nrep_h = 0; double total_h_ns = 0.0;
  while (nrep_h < 3 || total_h_ns < TMIN_SEC * 1e9) {
    auto s = std::chrono::high_resolution_clock::now();
    ref_gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', M, N, K, alpha, A.get(), M,
                         B.get(), K, beta, C_hybrid.get(), M, Ac.get(), Bc.get(),
                         mc_opt, nc_opt, kc_opt, MR, NR, rvv_ukr_adapter);
    auto e = std::chrono::high_resolution_clock::now();
    total_h_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    ++nrep_h;
  }
  const double gflops_h = (2.0 * M * N * K) / (total_h_ns / nrep_h);

  // -- Unpacked --
  int nrep_u = 0; double total_u_ns = 0.0;
  while (nrep_u < 3 || total_u_ns < TMIN_SEC * 1e9) {
    auto s = std::chrono::high_resolution_clock::now();
    gemm_unpacked_rvv(M, N, K, alpha, A.get(), M, B_row.get(), N, beta,
                     C_unpacked.get(), M);
    auto e = std::chrono::high_resolution_clock::now();
    total_u_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    ++nrep_u;

  }
  const double gflops_u = (2.0 * M * N * K) / (total_u_ns / nrep_u);

  // =========================================================================
  // 3. Output — always printed; error shown regardless of pass/fail
  // =========================================================================
  std::cout << " ------------------------------------------------------------------------------------------" << std::endl;
  std::cout << " | " << label << ": [M=" << M << ", N=" << N << ", K=" << K << "]"
            << "  tol=" << std::scientific << std::setprecision(1) << tol << std::endl;
  std::cout << " | > Compiled  MC=" << std::setw(5) << MC
            << "  NC=" << std::setw(6) << NC
            << "  KC=" << std::setw(5) << KC
            << "  (static, from cache_config.h)" << std::endl;
  std::cout << " | > Analytical MC=" << std::setw(5) << mc_calc
            << "  NC=" << std::setw(6) << nc_calc
            << "  KC=" << std::setw(5) << kc_calc << std::endl;
            
  if (is_overridden) {
    std::cout << " | > OVERRIDE   MC=" << std::setw(5) << mc_opt
              << "  NC=" << std::setw(6) << nc_opt
              << "  KC=" << std::setw(5) << kc_opt
              << "  (Using Manual Configs)" << std::endl;
  }
  std::cout << " | > RVV Packing + RVV Compute       [" << MR << "x" << NR << "]  "
            << std::fixed << std::setprecision(2) << gflops_p << " GFLOPS"
            << "  | MaxErr: " << std::scientific << std::setprecision(2) << err_packed
            << "  " << (pass_p ? "✅ PASS" : "⚠️  WARN") << "  (reps=" << nrep_p << ")" << std::endl;
  std::cout << " | > Scalar Packing + RVV Compute    [" << MR << "x" << NR << "]  "
            << std::fixed << std::setprecision(2) << gflops_h << " GFLOPS"
            << "  | MaxErr: " << std::scientific << std::setprecision(2) << err_hybrid
            << "  " << (pass_h ? "✅ PASS" : "⚠️  WARN") << "  (reps=" << nrep_h << ")" << std::endl;
  std::cout << " | > No Packing + RVV Compute        [" << MR << "x" << NR << "]  "
            << std::fixed << std::setprecision(2) << gflops_u << " GFLOPS"
            << "  | MaxErr: " << std::scientific << std::setprecision(2) << err_unpacked
            << "  " << (pass_u ? "✅ PASS" : "⚠️  WARN") << "  (reps=" << nrep_u << ")" << std::endl;
  std::cout << " ------------------------------------------------------------------------------------------" << std::endl;

  return pass_p && pass_h && pass_u;
}

// =========================================================================
// 4. CNN Model Runner
// =========================================================================

bool run_model(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "❌ Error: Could not open model file: " << filepath << std::endl;
    return false;
  }

  std::cout << "\n >>> Dataset: " << filepath << std::endl;
  std::cout << " ==========================================================================================" << std::endl;
  std::cout << " |                             GEMM_BLIS_XDSL STATIC KERNEL DRIVER                        |" << std::endl;
  std::cout << " ==========================================================================================" << std::endl;
  std::cout << " | [*] Lib Dir      :  tests/gemm_blis_xdsl/kernels/" << std::endl;
  std::cout << " | [*] CFLAGS       :  -O3 -mcmodel=medany -march=rv64gcv" << std::endl;
  std::cout << " | [*] Dataset      :  " << filepath << std::endl;
  std::cout << " | [*] Analytical   :  MC=" << MC << " NC=" << NC << " KC=" << KC << std::endl;
  std::cout << " | [*] Tmin         :  3.00" << std::endl;
  std::cout << " ------------------------------------------------------------------------------------------" << std::endl;

  std::string line;
  bool all_pass = true;
  int layer_idx = 1;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::stringstream ss(line);
    std::string layer_label;
    int m, n, k;
    
    // Read the label as a string first
    if (!(ss >> layer_label >> m >> n >> k))
      continue;

    // Read optional per-layer cache configs if they exist on the line
    int mc_over = 0, nc_over = 0, kc_over = 0;
    ss >> mc_over >> nc_over >> kc_over;

    char label[128];
    snprintf(label, sizeof(label), "Layer %d: %s", layer_idx++, layer_label.c_str());
    all_pass &= run_test(m, n, k, 1.0f, 1.0f, label, mc_over, nc_over, kc_over);
  }

  return all_pass;
}

// =========================================================================
// 5. Main Driver
// =========================================================================

int main(int argc, char **argv) {
  // Initialize dispatch matrix
  ukr_matrix = allocateMatrix();
  fillMatrix(ukr_matrix);

  bool all_pass = true;

  if (argc > 1) {
    // Run layers from the specified model file
    all_pass = run_model(argv[1]);
  } else {
    // Default tests if no model file provided
    std::cout << "No model file provided. Running default benchmarks..." << std::endl;
    
    // Test 1: Single panel (b1 mode)
    all_pass &= run_test(256, 256, 256, 1.0f, 1.0f, "Single-panel (beta=1)");

    // Test 2: Multi-panel (M > MC) (b1 mode)
    all_pass &= run_test(512, 512, 512, 1.0f, 1.0f, "Multi-panel (beta=1)");

    // Test 3: Multi-panel (b0 mode)
    all_pass &= run_test(512, 512, 512, 1.0f, 0.0f, "Multi-panel (beta=0)");
  }

  // Clean up dispatch matrix
  freeMatrix(ukr_matrix);

  return all_pass ? 0 : 1;
}
