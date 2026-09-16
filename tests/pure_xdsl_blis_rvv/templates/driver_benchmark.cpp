#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <tuple>
#include <cstdint>

#include "generated_kernels.h"

#if defined(DTYPE_INT8)
    typedef int8_t A_TYPE;
    typedef int8_t B_TYPE;
    typedef int32_t C_TYPE;
    #define A_CANARY ((A_TYPE)-123)
    #define B_CANARY ((B_TYPE)-123)
    #define C_CANARY ((C_TYPE)-123456)
    #define GET_ERR_TOL(k) 0.0
#else
    typedef float A_TYPE;
    typedef float B_TYPE;
    typedef float C_TYPE;
    #define A_CANARY -123456.0f
    #define B_CANARY -123456.0f
    #define C_CANARY -123456.0f
    #define GET_ERR_TOL(k) (std::max(1e-5, (double)(k) * 1e-6))
#endif

extern "C" {
#include "../../modelLevel/model_level.h"
}

using namespace std;

// Generate random data
template <typename T>
void fill_random(T* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
#if defined(DTYPE_INT8)
        ptr[i] = static_cast<T>((rand() % 10) - 5);
#else
        ptr[i] = static_cast<T>(rand()) / RAND_MAX;
#endif
    }
}

// Aligned allocation
template <typename T>
T* alloc_aligned(size_t size_elems) {
    size_t bytes = size_elems * sizeof(T);
    // Size must be a multiple of alignment for aligned_alloc
    size_t alloc_bytes = (bytes + 63) & ~63;
    void* ptr = aligned_alloc(64, alloc_bytes);
    if (!ptr) {
        cerr << "Memory allocation failed" << endl;
        exit(1);
    }
    return static_cast<T*>(ptr);
}

// ----------------------------------------------------------------------------
// LEGACY REFERENCE GEMM
// ----------------------------------------------------------------------------
#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]

void gemm_ref( char orderA, char orderB, char orderC, 
           char transA, char transB, 
           size_t m, size_t n, size_t k, 
           float alpha, A_TYPE *A, int ldA, 
                        B_TYPE *B, int ldB, 
           float beta,  C_TYPE *C, int ldC ){

   size_t    ic, jc, i, j, p;
   C_TYPE  zero = 0, one = 1, tmp;

   // Quick return if possible
  if ( (m==0)||(n==0) || (((alpha==zero) || (k==0)) && (beta==one)) )
    return;

  // Quick gemm if alpha==0.0
  if ( alpha==zero ) {
    if ( beta==zero )
      if ( orderC=='C' ) {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Ccol(ic,jc) = 0;
      }
      else {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Crow(ic,jc) = 0;
      }
    else
      if ( orderC=='C' ) {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Ccol(ic,jc) = beta*Ccol(ic,jc);
      }
      else {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Crow(ic,jc) = beta*Crow(ic,jc);
      }
    return;
  }

  if ( (transA=='N')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='N')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
}

// ----------------------------------------------------------------------------
// BENCHMARK MAIN
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    bool force_scalar_pack = (getenv("USE_SCALAR_PACK") != NULL);
    string dataset_file = "";
    
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--scalar-pack") {
            force_scalar_pack = true;
        } else {
            dataset_file = argv[i];
        }
    }

    if (force_scalar_pack) {
        cout << "[INFO] Scalar pack flag detected! Forcing scalar packing for benchmark runs." << endl;
    }
    if (dataset_file == "") {
        cerr << "Usage: " << argv[0] << " <dataset_file.dat> [--scalar-pack]" << endl;
        return 1;
    }

    // Ensure logs directory exists (handled by Makefile)
    string raw_log_name = string("logs/raw_") + CONFIG_DATATYPE + "_" + CONFIG_METHOD + ".txt";
    bool raw_exists = false;
    {
        ifstream check(raw_log_name);
        if (check.good() && check.peek() != std::ifstream::traits_type::eof()) {
            raw_exists = true;
        }
    }
    ofstream raw_log(raw_log_name, ios_base::app);
    if (!raw_log.is_open()) {
        cerr << "Warning: Could not open " << raw_log_name << " for writing." << endl;
    } else if (!raw_exists) {
        raw_log << "Layer,M,N,K,Variant,Status,KerGOPs_Harmonic,KerGOPs_Min,KerGOPs_P5,KerGOPs_P95,KerGOPs_Max,KerGOPs_IQR,KerGOPs_Mean,KerGOPs_Std,KerGOPs_Median,RVV_Pack_GBs,Scl_Pack_GBs,RVV_Time_ms,Ref_Time_ms\n";
    }

    string summary_log_name = string("logs/summary_") + CONFIG_DATATYPE + "_" + CONFIG_METHOD + ".txt";
    ofstream summary_log(summary_log_name, ios_base::app);
    if (!summary_log.is_open()) {
        cerr << "Warning: Could not open " << summary_log_name << " for writing." << endl;
    }

    ifstream dataset_stream(dataset_file);
    if (!dataset_stream.is_open()) {
        cerr << "Error: Could not open dataset file " << dataset_file << endl;
        return 1;
    }
    cout << "\n============================================================" << endl;
    cout << " Benchmark Config: Arch=" << CONFIG_ARCH 
         << " | Datatype=" << CONFIG_DATATYPE 
         << " | Method=" << CONFIG_METHOD 
         << " | VLEN=" << CONFIG_VLEN << endl;
    cout << " Dataset: " << dataset_file << endl;
    cout << "============================================================\n" << endl;

    cout << " Metric Explanations:\n";
    cout << "   - p5 / p95 : 5th and 95th percentiles (ignores top 5% lucky spikes and bottom 5% OS interrupts)\n";
    cout << "   - iqr      : Interquartile Range (P75 - P25), shows how tight the middle 50% of runs are clustered\n";
    cout << "   - s        : Standard deviation of Kernel GOPs across all runs\n";
    cout << "   - CV%      : Coefficient of Variation (s / median). >15% indicates severe performance instability\n";
    cout << "   - mx/md    : Max / Median ratio. >1.25 indicates extreme cache sensitivity (thrashing)\n";
    cout << "\n";

    cout << "  Layer Name (M x N x K)       | Unroll | Median GOPs [min, p5, p95, max, iqr, s, CV%, mx/md] | RVV Pack | Scl Pack | IPC   | Status\n";
    cout << "----------------------------------------------------------------------------------------------------------------------------------" << endl;


    string line;
    while (getline(dataset_stream, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        string label;
        int m, n, k;
        if (!(iss >> label >> m >> n >> k)) {
            continue; 
        }

        int lda = m, ldb = k, ldc = m;
        
        // Over-allocate buffers by 32 in each dimension to prevent segfaults/heap corruption.
        // The simple pure_xdsl macrokernel does not have edge-case handling for un-padded fringes,
        // so it will unconditionally write out of bounds up to the next MR/NR multiple.
        size_t pad_m = m + 32;
        size_t pad_n = n + 32;
        size_t pad_k = k + 32;
        A_TYPE* A = alloc_aligned<A_TYPE>(pad_m * pad_k);
        B_TYPE* B = alloc_aligned<B_TYPE>(pad_k * pad_n);
        C_TYPE* C = alloc_aligned<C_TYPE>(pad_m * pad_n);
        C_TYPE* Cg = alloc_aligned<C_TYPE>(pad_m * pad_n);
        
        fill_random(A, lda * k);
        fill_random(B, ldb * n);
        
        float alpha = 1.0f;
        float beta = 0.0f;

        string ref_time_str = "N/A";
        double ref_time_val_global = 0.0;
        string dim_str = to_string(m) + "x" + to_string(n) + "x" + to_string(k);
        // Bounded reference check (skip if too massive)
#ifdef SKIP_REF_RUN
        bool do_ref = false;
#else
        bool do_ref = true;
#endif
        if (do_ref) {
            cout << "\r" << "  [" << label << " (" << dim_str << ")] Running scalar reference ... " << flush;
            for (size_t i = 0; i < (size_t)(ldc * n); ++i) Cg[i] = 0;
            auto ref_start = chrono::high_resolution_clock::now();
            gemm_ref('C', 'C', 'C', 'N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, Cg, ldc);
            auto ref_end = chrono::high_resolution_clock::now();
            chrono::duration<double> ref_diff = ref_end - ref_start;
            ref_time_val_global = ref_diff.count() * 1000.0;
            
            stringstream ss;
            ss << fixed << setprecision(4) << ref_time_val_global;
            ref_time_str = ss.str();
        }

#ifndef CONFIG_ITERATIONS
        const int N_REPS = 5;
#else
        const int N_REPS = CONFIG_ITERATIONS;
#endif

        struct VariantResult {
            string name;
            string status;
            double ker_gflops;
            double ker_gflops_min;
            double ker_gflops_p5;
            double ker_gflops_p95;
            double ker_gflops_max;
            double ker_gflops_iqr;
            double ker_gflops_mean;
            double ker_gflops_std;
            double ker_gflops_median;
            double rvv_bw;
            double scl_bw;
            double rvv_time_ms;
            double ref_time_ms;
            double sample_rvv;
            double sample_ref;
            int sample_i;
            int sample_j;
            int mc;
            int nc;
            int kc;
        };
        vector<VariantResult> variant_results;

        for (int v = 0; v < num_variants; ++v) {
            const auto& variant = generated_variants[v];

            int mc, nc, kc;
            get_optim_mc_nc_kc(sizeof(float), m, n, k, variant.mr, variant.nr, &mc, &nc, &kc);

            // Debug/mitigation override for issue #1 (B3_M7 packing instability): apply a
            // per-shape nc override from config.json (matched by longest label prefix),
            // without changing get_optim_mc_nc_kc's general heuristic for any other shape.
            {
                size_t best_len = 0;
                int best_nc = nc;
                for (size_t oi = 0; oi < CONFIG_NC_OVERRIDES_COUNT; ++oi) {
                    const auto& ov = CONFIG_NC_OVERRIDES[oi];
                    size_t plen = std::strlen(ov.prefix);
                    if (plen > best_len && label.compare(0, plen, ov.prefix) == 0) {
                        best_len = plen;
                        best_nc = (n < ov.nc) ? n : ov.nc;
                    }
                }
                nc = best_nc;
            }

            // Ad hoc override for quick interactive experiments; takes priority over config.json.
            if (const char* forced_nc = std::getenv("RVV_FORCE_NC")) {
                int nc_req = std::atoi(forced_nc);
                nc = (n < nc_req) ? n : nc_req;
            }

            int kc_pad = kc;
            if (string(CONFIG_METHOD).find("DOT") != string::npos) {
                kc_pad = ((kc + 3) / 4) * 4;
            }

            size_t padded_mc = ((mc + variant.mr - 1) / variant.mr) * variant.mr;
            size_t padded_nc = ((nc + variant.nr - 1) / variant.nr) * variant.nr;
            // Match reference design memory allocation sizes
            int32_t ac_alloc_size = (variant.mr + mc) * (variant.mr + kc);
            int32_t bc_alloc_size = (variant.nr + kc) * (variant.nr + nc);
            A_TYPE* Ac = alloc_aligned<A_TYPE>(ac_alloc_size);
            B_TYPE* Bc = alloc_aligned<B_TYPE>(bc_alloc_size);
     // Re-initialize C for this variant
            for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0;

            // Warmup: run multiple times to bring A, B, Ac, Bc hot into cache
            // A single warmup run is insufficient for large working sets
            for (int w = 0; w < 5; ++w) {
                double dump_pt=0, dump_kt=0; uint64_t dump_pb=0;
                variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr, &dump_pt, &dump_kt, &dump_pb, force_scalar_pack);
            }

            string status = "SKIP";
            double sample_rvv = 0.0;
            double sample_ref = 0.0;
            int sample_i = rand() % m;
            int sample_j = rand() % n;

            if (do_ref) {
                sample_rvv = C[sample_j * ldc + sample_i];
                sample_ref = Cg[sample_j * ldc + sample_i];

                double error = 0.0;
                double nrm = 0.0;
                double tmp = 0.0;
                int mismatched = 0;
                // Legacy error computation
                for (size_t j = 0; j < (size_t)n; ++j) {
                    for (size_t i = 0; i < (size_t)m; ++i) {
                        double ref_val = (double) Cg[j * ldc + i];
                        nrm += ref_val * ref_val;
                        tmp = (double) std::abs(ref_val - C[j * ldc + i]);
                        error += tmp * tmp;
                        
                        // Track first few mismatches for debugging
                        if (tmp > 1e-4 && mismatched < 5) {
                            if (mismatched == 0) cout << endl << "  [MISMATCH DETECTED] " << variant.name << endl;
                            cout << "    C[" << i << "," << j << "] -> Ref: " << Cg[j * ldc + i] << " | RVV: " << C[j * ldc + i] << " | Diff: " << tmp << endl;
                            mismatched++;
                        }
                    }
                }
                if (nrm != 0.0) {
                    error = sqrt(error) / sqrt(nrm);
                } else {
                    error = sqrt(error);
                }
                
                double tol = GET_ERR_TOL(k);
                if (error > tol) status = "FAIL";
                else status = "PASS";
            }

            string current_ref_time = ref_time_str;
            string label_str = label;
            string var_str = string("`") + variant.name + "`";
            string status_str;
            if (status == "PASS") status_str = "✅ PASS";
            else if (status == "SKIP") status_str = "⏭️ SKIP";
            else status_str = "❌ FAIL";

            cout << "\r" << "  [" << label << " (" << dim_str << ")] Testing variant " << (v+1) << "/" << num_variants << " : " << left << setw(15) << variant.name << flush;
            
            double ker_gflops_val = 0.0;
            double ker_gflops_min = 0.0;
            double ker_gflops_p5 = 0.0;
            double ker_gflops_p95 = 0.0;
            double ker_gflops_max = 0.0;
            double ker_gflops_iqr = 0.0;
            double ker_gflops_mean = 0.0;
            double ker_gflops_std = 0.0;
            double ker_gflops_median = 0.0;
            double rvv_bw_val = 0.0;
            double scl_bw_val = 0.0;
            double rvv_time_val = 0.0;
            double ref_time_val = ref_time_val_global;

            if (status != "FAIL") {
                // Benchmark Scalar Packing
                // Reset C to zero so each benchmark starts from a consistent state
                for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0;
                double total_scalar_pack_time = 0.0;
                uint64_t total_pack_bytes = 0;
                for (int r = 0; r < N_REPS; ++r) {
                    double pt = 0.0, kt = 0.0; uint64_t pb = 0;
                    variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr, &pt, &kt, &pb, true);
                    total_scalar_pack_time += pt;
                    total_pack_bytes += pb;
                }
                scl_bw_val = (double)total_pack_bytes / (total_scalar_pack_time * 1e9);

                // Benchmark RVV
                // Reset C again before RVV loop so it starts from the same clean state
                for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0;
                double total_rvv_pack_time = 0.0;
                double total_kernel_time = 0.0;
                total_pack_bytes = 0;
                std::vector<double> rep_kt(N_REPS);

                auto start = chrono::high_resolution_clock::now();
                for (int r = 0; r < N_REPS; ++r) {
                    double pt = 0.0, kt = 0.0; uint64_t pb = 0;
                    variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr, &pt, &kt, &pb, force_scalar_pack);
                    total_rvv_pack_time += pt;
                    total_kernel_time += kt;
                    total_pack_bytes += pb;
                    rep_kt[r] = kt;
                }
                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> diff = end - start;
                double avg_time = diff.count() / N_REPS;
                
                double num_ops = 2.0 * m * n * k;
                ker_gflops_val = (num_ops * N_REPS) / (total_kernel_time * 1e9);
                
                // Compute per-rep stats
                std::vector<double> rep_gops(N_REPS);
                double sum_gops = 0.0;
                for (int r = 0; r < N_REPS; ++r) {
                    rep_gops[r] = num_ops / (rep_kt[r] * 1e9);
                    sum_gops += rep_gops[r];
                }
                std::sort(rep_gops.begin(), rep_gops.end());
                ker_gflops_min = rep_gops.front();
                ker_gflops_p5 = rep_gops[(int)(0.05 * N_REPS)];
                ker_gflops_median = rep_gops[N_REPS / 2];
                ker_gflops_p95 = rep_gops[(int)(0.95 * N_REPS)];
                ker_gflops_max = rep_gops.back();
                ker_gflops_iqr = rep_gops[(int)(0.75 * N_REPS)] - rep_gops[(int)(0.25 * N_REPS)];
                
                double mean_gops = sum_gops / N_REPS;
                ker_gflops_mean = mean_gops;
                
                double sq_sum = 0.0;
                for (int r = 0; r < N_REPS; ++r) {
                    sq_sum += (rep_gops[r] - mean_gops) * (rep_gops[r] - mean_gops);
                }
                ker_gflops_std = std::sqrt(sq_sum / N_REPS);

                rvv_bw_val = (double)total_pack_bytes / (total_rvv_pack_time * 1e9);
                rvv_time_val = avg_time * 1000.0;
            }
            
            variant_results.push_back({string(variant.name), status, ker_gflops_val, ker_gflops_min, ker_gflops_p5, ker_gflops_p95, ker_gflops_max, ker_gflops_iqr, ker_gflops_mean, ker_gflops_std, ker_gflops_median, rvv_bw_val, scl_bw_val, rvv_time_val, ref_time_val, sample_rvv, sample_ref, sample_i, sample_j, mc, nc, kc});

            if (raw_log.is_open()) {
                raw_log << label_str << "," << m << "," << n << "," << k << "," 
                        << variant.name << "," << status << "," 
                        << ker_gflops_val << "," 
                        << ker_gflops_min << "," << ker_gflops_p5 << "," 
                        << ker_gflops_p95 << "," << ker_gflops_max << ","
                        << ker_gflops_iqr << ","
                        << ker_gflops_mean << "," << ker_gflops_std << ","
                        << ker_gflops_median << ","
                        << rvv_bw_val << "," << scl_bw_val << ","
                        << rvv_time_val << "," << ref_time_val << endl;
            }
            
            free(Ac);
            free(Bc);
        }

        // Print Top 3 for this problem
        cout << "\r                                                                               \r";
        sort(variant_results.begin(), variant_results.end(), [](const VariantResult& a, const VariantResult& b) {
            if (a.status == "PASS" && b.status != "PASS") return true;
            if (a.status != "PASS" && b.status == "PASS") return false;
            return a.ker_gflops > b.ker_gflops;
        });

        string problem_str = label + " (" + dim_str + ")";
        stringstream ss;
        ss << "▶ " << left << setw(22) << problem_str << " | ";
        if (!variant_results.empty()) {
            auto& vr = variant_results[0];
            
            // Simplify "MR16_NR15" to "16x15"
            string short_name = vr.name;
            size_t pos = short_name.find("MR");
            if (pos != string::npos) short_name.replace(pos, 2, "");
            pos = short_name.find("_NR");
            if (pos != string::npos) short_name.replace(pos, 3, "x");

            double cv_pct = (vr.ker_gflops_median > 0) ? (vr.ker_gflops_std / vr.ker_gflops_median) * 100.0 : 0.0;
            double max_md = (vr.ker_gflops_median > 0) ? (vr.ker_gflops_max / vr.ker_gflops_median) : 0.0;

            ss << left << setw(6) << short_name << " " 
                 << right << setw(5) << fixed << setprecision(2) << vr.ker_gflops_median << " KerGOPs "
                 << "[min:" << fixed << setprecision(1) << vr.ker_gflops_min 
                 << ", p5:" << vr.ker_gflops_p5
                 << ", p95:" << vr.ker_gflops_p95
                 << ", max:" << vr.ker_gflops_max 
                 << ", iqr:" << vr.ker_gflops_iqr
                 << ", s:" << vr.ker_gflops_std 
                 << ", CV:" << fixed << setprecision(0) << cv_pct << "%"
                 << ", mx/md:" << fixed << setprecision(2) << max_md << "] | "
                 << right << setw(4) << fixed << setprecision(1) << vr.rvv_bw << " GB/s Pk | "
                 << right << setw(4) << fixed << setprecision(1) << vr.scl_bw << " GB/s Scl | ";
            if (vr.status == "PASS") {
                ss << "✅ C[" << vr.sample_i << "," << vr.sample_j 
                     << "] " << fixed << setprecision(2) << vr.sample_ref 
                     << "==" << vr.sample_rvv;
            } else if (vr.status == "SKIP") {
                ss << "⏭️ SKIP";
            } else {
                ss << "❌ " << vr.status;
            }
            ss << " (mc:" << vr.mc << " nc:" << vr.nc << " kc:" << vr.kc << ")";
        }
        string output_line = ss.str();
        cout << output_line << endl;
        
        if (summary_log.is_open()) {
            summary_log << output_line << "\n";
        }

        free(A);
        free(B);
        free(C);
        free(Cg);
    }

    return 0;
}
