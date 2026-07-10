#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "generated_kernels.h"

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

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_file.dat>" << endl;
        return 1;
    }

    string dataset_path = argv[1];
    ifstream file(dataset_path);
    if (!file.is_open()) {
        cerr << "Error opening file: " << dataset_path << endl;
        return 1;
    }

    cout << "M;N;K;Variant;GFLOPS;C[0]" << endl;

    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        // The dat file format has a label column first, then M N K
        istringstream iss(line);
        string label;
        int m, n, k;
        if (!(iss >> label >> m >> n >> k)) {
            continue; 
        }

        // Allocate memory with 64-byte alignment for RVV
        // lda = m, ldb = k, ldc = m (Column major)
        int lda = m, ldb = k, ldc = m;
        float* A = alloc_aligned(lda * k);
        float* B = alloc_aligned(ldb * n);
        float* C = alloc_aligned(ldc * n);
        
        fill_random(A, lda * k);
        fill_random(B, ldb * n);
        for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0.0f;

        float alpha = 1.0f;
        float beta = 0.0f;

        const int N_REPS = 5;

        // Iterate through generated variants
        for (int v = 0; v < num_variants; ++v) {
            const auto& variant = generated_variants[v];

            // Allocate packing buffers based on cache config
            size_t padded_mc = ((variant.mc + variant.mr - 1) / variant.mr) * variant.mr;
            size_t padded_nc = ((variant.nc + variant.nr - 1) / variant.nr) * variant.nr;
            float* Ac = alloc_aligned(padded_mc * variant.kc);
            float* Bc = alloc_aligned(variant.kc * padded_nc);

            // Warmup
            cout << label << ";" << m << ";" << n << ";" << k << ";" << variant.name << ";" << flush;
            variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc);

            // Benchmark
            auto start = chrono::high_resolution_clock::now();
            for (int r = 0; r < N_REPS; ++r) {
                variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc);
            }
            auto end = chrono::high_resolution_clock::now();

            chrono::duration<double> diff = end - start;
            double avg_time = diff.count() / N_REPS;
            double gflops = (2.0 * m * n * k) / (avg_time * 1e9);

            cout << fixed << setprecision(4) << gflops << ";" << C[0] << endl;
        }
    }

    return 0;
}
