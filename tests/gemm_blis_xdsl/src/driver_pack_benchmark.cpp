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

// Include statically generated cache configuration for MR_OPT / NR_OPT
#include "gen/cache_config.h"
#include "gemm_macro_dispatch.h"

// Hardcode MR and NR to what the macro-kernel uses
static constexpr int MR = MR_OPT;
static constexpr int NR = NR_OPT;

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

// =========================================================================
// 1. Scalar Reference Implementations
// Ported from xdsl_sweep_driver/xdsl_gemm_blis.c
// =========================================================================

void ref_pack_CB(char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR) {
    int i, j, jj, k, nr;
    k = 0;
    if (((transM=='N')&&(orderM=='C')) || ((transM=='T')&&(orderM=='R'))) {
        for (j=0; j<nc; j+=RR) {
            k = j*mc;
            nr = std::min(nc-j, RR);
            for (i=0; i<mc; i++) {
                for (jj=0; jj<nr; jj++) {
                    Mc[k] = Mcol(i,j+jj);
                    k++;
                }
                k += (RR-nr);
            }
        }
    } else {
        for (j=0; j<nc; j+=RR) {
            k = j*mc;
            nr = std::min(nc-j, RR);
            for (i=0; i<mc; i++) {
                for (jj=0; jj<nr; jj++) {
                    Mc[k] = Mcol(j+jj,i);
                    k++;
                }
                k += (RR-nr);
            }
        }
    }
}

void ref_pack_RB(char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR) {
    int i, j, ii, k, rr;
    if (((transM=='N')&&(orderM=='C')) || ((transM=='T')&&(orderM=='R'))) {
        for (i=0; i<mc; i+=RR) {
            k = i*nc;
            rr = std::min(mc-i, RR);
            for (j=0; j<nc; j++) {
                for (ii=0; ii<rr; ii++) {
                    Mc[k] = Mcol(i+ii, j);
                    k++;
                }
                k += (RR-rr);
            }
        }
    } else {
        for (i=0; i<mc; i+=RR) {
            k = i*nc;
            rr = std::min(mc-i, RR);
            for (j=0; j<nc; j++) {
                for (ii=0; ii<rr; ii++) {
                    Mc[k] = Mcol(j, i+ii);
                    k++;
                }
                k += (RR-rr);
            }
        }
    }
}

// Dummy Scalar Microkernel
void dummy_scalar_microkernel(int mr, int nr, int kc, const void *alpha,
                              const void *a, const void *b, const void *beta,
                              void *c, int inc_c, int ld_c, const void *aux,
                              const void *cntx) {
    // Empty compute!
}

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]

// Reference 5-loop scalar macro-kernel
void ref_gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
                           char transA, char transB, 
                           size_t m, size_t n, size_t k, 
                           float alpha, float *A, size_t ldA, 
                           float *B, size_t ldB, 
                           float beta,  float *C, size_t ldC, 
                           float *Ac, float *Bc, 
                           size_t MC, size_t NC, size_t KC,
                           int MR_arg, int NR_arg,
                           gemm_ukr_ft gemm_kernel) {
  size_t ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float zero = 0.0, one = 1.0, betaI; 
  float *Aptr, *Bptr, *Cptr;
   
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) ) return;
  
  for ( jc=0; jc<n; jc+=NC ) {
    nc = std::min(n-jc, NC); 
    for ( pc=0; pc<k; pc+=KC ) {
      kc = std::min(k-pc, KC); 
      
      if ( (transB=='N')&&(orderB=='C') )      Bptr = &Bcol(pc,jc);
      else if ( (transB=='N')&&(orderB=='R') ) Bptr = &Brow(pc,jc);
      else if ( (transB=='T')&&(orderB=='C') ) Bptr = &Bcol(jc,pc);
      else                                     Bptr = &Brow(jc,pc);
      
      ref_pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR_arg);
      
      if ( pc==0 ) betaI = beta;
      else         betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = std::min(m-ic, MC); 
        
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

// =========================================================================
// 2. RVV Empty Microkernel
// =========================================================================

extern "C" {
    // Dummy microkernel to satisfy linker and bypass compute in RVV macro
    void gemm_microkernel_v(int mr, int nr, int kc, float alpha, float *Ac_ptr,
                            float *Bc_ptr, float betaI, float *Cptr, int ldC) {
        // Empty compute!
    }
}

// =========================================================================
// 3. Analytical Cache Model
// =========================================================================

static int cache_model_level(bool isL3, int NL, int CL, int WL, int ds, int m, int n) {
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

static void per_layer_mc_nc_kc(int M, int N, int K, int *mc_opt, int *nc_opt, int *kc_opt) {
  const int NL1=128,CL1=64,WL1=4;
  const int NL2=1024,CL2=64,WL2=8;
  const int NL3=4096,CL3=64,WL3=16;
  const int ds = 4;
  int kc = cache_model_level(false, NL1,CL1,WL1, ds, MR, NR);
  kc = std::min(K, kc);
  int mc = cache_model_level(false, NL2,CL2,WL2, ds, kc, NR);
  mc = std::min(M, mc);
  int nc = cache_model_level(true,  NL3,CL3,WL3, ds, kc, mc);
  nc = std::min(N, nc);
  *kc_opt = kc;  *mc_opt = mc;  *nc_opt = nc;
}

// =========================================================================
// 4. Benchmarking Infrastructure
// =========================================================================

static float* aligned_float_alloc(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 32, count * sizeof(float)) != 0) {
        std::cerr << "aligned alloc failed\n"; std::exit(1);
    }
    std::memset(ptr, 0, count * sizeof(float));
    return static_cast<float*>(ptr);
}

struct PackResult {
    double scalar_time_ms;
    double rvv_time_ms;
    double speedup;
};

PackResult benchmark_macro_overhead(int M, int N, int K) {
    auto A = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * K), free);
    auto B = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(K * N), free);
    auto C = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(M * N), free);

    int mc_opt, nc_opt, kc_opt;
    per_layer_mc_nc_kc(M, N, K, &mc_opt, &nc_opt, &kc_opt);

    // Allow manual injection from bash environment
    if (const char* env_mc = std::getenv("OVERRIDE_MC")) mc_opt = std::stoi(env_mc);
    if (const char* env_nc = std::getenv("OVERRIDE_NC")) nc_opt = std::stoi(env_nc);
    if (const char* env_kc = std::getenv("OVERRIDE_KC")) kc_opt = std::stoi(env_kc);

    const int KR = MR;
    const size_t ac_size = std::max((size_t)(mc_opt * kc_opt), (size_t)(MR + mc_opt) * (KR + kc_opt));
    const size_t bc_size = std::max((size_t)(nc_opt * kc_opt), (size_t)(KR + kc_opt) * (NR + nc_opt));
    
    auto Ac = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(ac_size), free);
    auto Bc = std::unique_ptr<float[], decltype(&free)>(aligned_float_alloc(bc_size), free);

    const double TMIN_SEC = 0.2; // 200ms
    float alpha = 1.0f, beta = 0.0f;

    // Benchmark Scalar Macro-Kernel Overhead
    int nrep_s = 0; double total_s_ns = 0.0;
    while (nrep_s < 3 || total_s_ns < TMIN_SEC * 1e9) {
        auto s = std::chrono::high_resolution_clock::now();
        ref_gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', M, N, K, alpha, A.get(), M, B.get(), K, beta, C.get(), M, 
                             Ac.get(), Bc.get(), mc_opt, nc_opt, kc_opt, MR, NR, dummy_scalar_microkernel);
        auto e = std::chrono::high_resolution_clock::now();
        total_s_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
        nrep_s++;
    }

    // Benchmark RVV Macro-Kernel Overhead
    int nrep_r = 0; double total_r_ns = 0.0;
    while (nrep_r < 3 || total_r_ns < TMIN_SEC * 1e9) {
        auto s = std::chrono::high_resolution_clock::now();
        gemm_blis_v_CN_CN_C_fp32(M, N, K, alpha, A.get(), M, B.get(), K, beta, C.get(), M, Ac.get(), Bc.get());
        auto e = std::chrono::high_resolution_clock::now();
        total_r_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
        nrep_r++;
    }

    PackResult res;
    res.scalar_time_ms = (total_s_ns / nrep_s) / 1e6;
    res.rvv_time_ms = (total_r_ns / nrep_r) / 1e6;
    res.speedup = res.scalar_time_ms / res.rvv_time_ms;
    return res;
}

void print_row(const char* category, int m, int n, int k, PackResult r) {
    char size_str[40];
    std::snprintf(size_str, sizeof(size_str), "%d x %d x %d", m, n, k);
    std::printf(" | %-16s | %-18s | %16.2f | %13.2f | %7.2fx |\n", 
                category, size_str, r.scalar_time_ms, r.rvv_time_ms, r.speedup);
}

// =========================================================================
// 5. Main Execution
// =========================================================================

int main(int argc, char **argv) {
    std::printf("\n");
    std::printf(" ==========================================================================================\n");
    std::printf(" | MACRO-KERNEL OVERHEAD BENCHMARK (5 LOOPS + PACKING, NO COMPUTE)\n");
    std::printf(" | Hardware: RVV vs Scalar  [MR=%d, NR=%d]\n", MR, NR);
    std::printf(" ==========================================================================================\n");
    std::printf(" | %-16s | %-18s | %16s | %13s | %8s |\n", "Category", "Matrix (M x N x K)", "Scalar Time (ms)", "RVV Time (ms)", "Speedup");
    std::printf(" ------------------------------------------------------------------------------------------\n");

    struct Dim { const char* name; int m; int n; int k; };

    Dim sweeps[] = {
        {"Square (Tiny)",    32, 32, 32},
        {"Square (Small)",   64, 64, 64},
        {"Square (Med)",    256, 256, 256},
        {"Square (Large)",  512, 512, 512},
        {"Square (V.Large)",1024, 1024, 1024},
        {"Tall A",         1024, 64, 64},
        {"Wide B",           64, 1024, 64},
        {"Deep K",           64, 64, 1024},
        {"Transformer 1",  1024, 64, 1024},
        {"Transformer 2",   256, 256, 4096},
        {"Odd Dims",        127, 255, 333}
    };

    for (auto d : sweeps) print_row(d.name, d.m, d.n, d.k, benchmark_macro_overhead(d.m, d.n, d.k));

    // Phase C: Model Specific
    if (argc > 1) {
        std::string filename = argv[1];
        std::ifstream file(filename);
        if (file) {
            std::string line;
            if (std::getline(file, line)) { // skip header
                int layer = 1;
                while (std::getline(file, line)) { // process all layers
                    std::stringstream ss(line);
                    int M, N, K;
                    if (ss >> M >> N >> K) {
                        char name[32];
                        std::snprintf(name, sizeof(name), "Model Layer %d", layer++);
                        print_row(name, M, N, K, benchmark_macro_overhead(M, N, K));
                    }
                }
            }
        }
    }
    std::printf(" ------------------------------------------------------------------------------------------\n\n");

    return 0;
}
