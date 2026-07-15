#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <tuple>

#include "generated_kernels.h"

extern "C" {
#include "../../modelLevel/model_level.h"
}

using namespace std;

// Generate random data
void fill_random(float* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Aligned allocation
float* alloc_aligned(size_t size_floats) {
    size_t bytes = size_floats * sizeof(float);
    // Size must be a multiple of alignment for aligned_alloc
    size_t alloc_bytes = (bytes + 63) & ~63;
    void* ptr = aligned_alloc(64, alloc_bytes);
    if (!ptr) {
        cerr << "Memory allocation failed" << endl;
        exit(1);
    }
    return static_cast<float*>(ptr);
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
           float alpha, float *A, int ldA, 
                        float *B, int ldB, 
           float beta,  float *C, int ldC ){

   size_t    ic, jc, i, j, p;
   float  zero = 0.0f, one = 1.0f, tmp;

   // Quick return if possible
  if ( (m==0)||(n==0) || (((alpha==zero) || (k==0)) && (beta==one)) )
    return;

  // Quick gemm if alpha==0.0
  if ( alpha==zero ) {
    if ( beta==zero )
      if ( orderC=='C' ) {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Ccol(ic,jc) = 0.0;
      }
      else {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Crow(ic,jc) = 0.0;
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
        tmp = 0.0; 
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
        tmp = 0.0; 
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
        tmp = 0.0; 
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
        tmp = 0.0; 
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
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_file.dat>" << endl;
        return 1;
    }

    // Ensure logs directory exists (handled by Makefile)
    ofstream raw_log("logs/raw_results.txt", ios_base::app);
    if (!raw_log.is_open()) {
        cerr << "Warning: Could not open logs/raw_results.txt for writing." << endl;
    }

    ofstream best_log("logs/best_results.txt", ios_base::app);
    if (!best_log.is_open()) {
        cerr << "Warning: Could not open logs/best_results.txt for writing." << endl;
    } else {
        best_log.seekp(0, ios::end);
        if (best_log.tellp() == 0) {
            best_log << "Arch,Datatype,Method,VLEN,Label,M,N,K,Best_Kernel,GFLOPS,RVV_Time_ms,Ref_Time_ms\n";
        }
    }

    string dataset_path = argv[1];
    ifstream file(dataset_path);
    if (!file.is_open()) {
        cerr << "Error opening file: " << dataset_path << endl;
        return 1;
    }

    cout << "\n============================================================" << endl;
    cout << " Benchmark Config: Arch=" << CONFIG_ARCH 
         << " | Datatype=" << CONFIG_DATATYPE 
         << " | Method=" << CONFIG_METHOD 
         << " | VLEN=" << CONFIG_VLEN << endl;
    cout << " Dataset: " << dataset_path << endl;
    cout << "============================================================\n" << endl;


    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        string label;
        int m, n, k;
        if (!(iss >> label >> m >> n >> k)) {
            continue; 
        }

        int lda = m, ldb = k, ldc = m;
        float* A = alloc_aligned(lda * k);
        float* B = alloc_aligned(ldb * n);
        float* C = alloc_aligned(ldc * n);
        float* Cg = alloc_aligned(ldc * n);
        
        fill_random(A, lda * k);
        fill_random(B, ldb * n);
        
        float alpha = 1.0f;
        float beta = 0.0f;

        string ref_time_str = "N/A";
        double ref_time_val_global = 0.0;
        string dim_str = to_string(m) + "x" + to_string(n) + "x" + to_string(k);
        // Bounded reference check (skip if too massive)
        bool do_ref = true;
        if (do_ref) {
            cout << "\r" << "  [" << label << " (" << dim_str << ")] Running scalar reference ... " << flush;
            // Initialize Cg
            for (size_t i = 0; i < (size_t)(ldc * n); ++i) Cg[i] = 0.0f;
            auto ref_start = chrono::high_resolution_clock::now();
            gemm_ref('C', 'C', 'C', 'N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, Cg, ldc);
            auto ref_end = chrono::high_resolution_clock::now();
            chrono::duration<double> ref_diff = ref_end - ref_start;
            ref_time_val_global = ref_diff.count() * 1000.0;
            
            stringstream ss;
            ss << fixed << setprecision(4) << ref_time_val_global;
            ref_time_str = ss.str();
        }

        const int N_REPS = 5;

        struct VariantResult {
            string name;
            string status;
            double gflops;
            double rvv_time_ms;
            double ref_time_ms;
        };
        vector<VariantResult> variant_results;

        for (int v = 0; v < num_variants; ++v) {
            const auto& variant = generated_variants[v];

            int mc, nc, kc;
            get_optim_mc_nc_kc(sizeof(float), m, n, k, variant.mr, variant.nr, &mc, &nc, &kc);

            size_t padded_mc = ((mc + variant.mr - 1) / variant.mr) * variant.mr;
            size_t padded_nc = ((nc + variant.nr - 1) / variant.nr) * variant.nr;
            size_t ac_len = padded_mc * kc;
            size_t bc_len = kc * padded_nc;
            float* Ac_alloc = alloc_aligned(ac_len + 32);
            float* Bc_alloc = alloc_aligned(bc_len + 32);
            float* Ac = Ac_alloc + 16;
            float* Bc = Bc_alloc + 16;
            for (size_t i = 0; i < 16; ++i) {
                Ac_alloc[i] = -123456.0f;
                Ac_alloc[16 + ac_len + i] = -123456.0f;
                Bc_alloc[i] = -123456.0f;
                Bc_alloc[16 + bc_len + i] = -123456.0f;
            }
            if (string(variant.name) == "MR4_NR1") {
                cout << "MR4_NR1 allocations: Ac=" << Ac << " Bc=" << Bc << " C=" << C << " bc_len=" << bc_len << endl;
            }

            // Re-initialize C for this variant
            for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0.0f;

            // Warmup (also computes C)
            variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr);

            string status = "SKIP";
            if (do_ref) {
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
                
                double tol = std::max(1e-5, (double)k * 1e-6);
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
            
            double gflops_val = 0.0;
            double rvv_time_val = 0.0;
            double ref_time_val = ref_time_val_global;

            if (status != "FAIL") {
                // Benchmark
                auto start = chrono::high_resolution_clock::now();
                for (int r = 0; r < N_REPS; ++r) {
                    variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr);
                }
                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> diff = end - start;
                double avg_time = diff.count() / N_REPS;
                gflops_val = (2.0 * m * n * k) / (avg_time * 1e9);
                rvv_time_val = avg_time * 1000.0;
            }
            
            variant_results.push_back({string(variant.name), status, gflops_val, rvv_time_val, ref_time_val});

            if (raw_log.is_open()) {
                raw_log << label_str << "," << m << "," << n << "," << k << "," 
                        << variant.name << "," << status << "," 
                        << gflops_val << "," << rvv_time_val << "," << ref_time_val << endl;
            }

            // Check bounds after execution
            bool corrupted = false;
            for (size_t i = 0; i < 16; ++i) {
                if (Ac_alloc[i] != -123456.0f) { cout << "Ac front canary corrupted at " << i << "! Value: " << Ac_alloc[i] << endl; corrupted = true; }
                if (Ac_alloc[16 + ac_len + i] != -123456.0f) { cout << "Ac back canary corrupted at " << i << "! Value: " << Ac_alloc[16 + ac_len + i] << endl; corrupted = true; }
                if (Bc_alloc[i] != -123456.0f) { cout << "Bc front canary corrupted at " << i << "! Value: " << Bc_alloc[i] << endl; corrupted = true; }
                if (Bc_alloc[16 + bc_len + i] != -123456.0f) { cout << "Bc back canary corrupted at " << i << "! Value: " << Bc_alloc[16 + bc_len + i] << endl; corrupted = true; }
            }
            if (corrupted) {
                cout << "MEMORY CORRUPTION DETECTED in variant " << variant.name << ". Exiting." << endl;
                exit(1);
            }
            
            free(Ac_alloc);
            free(Bc_alloc);
        }

        // Print Top 3 for this problem
        cout << "\r                                                                               \r";
        sort(variant_results.begin(), variant_results.end(), [](const VariantResult& a, const VariantResult& b) {
            if (a.status == "PASS" && b.status != "PASS") return true;
            if (a.status != "PASS" && b.status == "PASS") return false;
            return a.gflops > b.gflops;
        });

        string problem_str = label + " (" + dim_str + ")";
        cout << "▶ " << left << setw(30) << problem_str << " | ";
        for (size_t i = 0; i < min((size_t)3, variant_results.size()); ++i) {
            auto& vr = variant_results[i];
            cout << "#" << (i+1) << ": " << left << setw(8) << vr.name << " " 
                 << fixed << setprecision(2) << vr.gflops << " GOPs  ";
            if (vr.status != "PASS") {
                cout << "[" << vr.status << "] ";
            }
        }
        cout << endl;

        if (!variant_results.empty() && best_log.is_open()) {
            auto& best = variant_results[0];
            best_log << CONFIG_ARCH << "," << CONFIG_DATATYPE << "," << CONFIG_METHOD << "," << CONFIG_VLEN << ","
                     << label << "," << m << "," << n << "," << k << "," 
                     << best.name << "," << best.gflops << "," 
                     << best.rvv_time_ms << "," << best.ref_time_ms << "\n";
        }

        free(A);
        free(B);
        free(C);
        free(Cg);
    }

    return 0;
}
