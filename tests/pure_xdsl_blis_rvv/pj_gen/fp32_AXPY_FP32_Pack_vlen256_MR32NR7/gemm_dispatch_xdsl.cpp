// CONFIG METHOD: AXPY_FP32_Pack
// DATATYPE: fp32
#include "gemm_headers.h"

extern "C" {

void gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc, size_t max_mr, size_t max_nr) {
    if (beta == (float)0) { gemm_microkernel_v_fp32_32x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); }
    else { gemm_microkernel_v_fp32_32x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); }
}

static inline void pack_RB_scalar(int32_t mc, int32_t nc, float* M, int32_t ldM, float* Mc, int32_t RR) {
    int32_t i, j, ii, k, rr;
    for (i = 0; i < mc; i += RR) {
        k = i * nc;
        rr = std::min(mc - i, RR);
        for (j = 0; j < nc; j++) {
            for (ii = 0; ii < rr; ii++) {
                Mc[k] = M[(j) * ldM + (i + ii)];
                k++;
            }
            k += (RR - rr);
        }
    }
}

static inline void pack_CB_scalar(int32_t mc, int32_t nc, float* M, int32_t ldM, float* Mc, int32_t RR) {
    int32_t i, j, jj, k, nr;
    for (j = 0; j < nc; j += RR) {
        k = j * mc;
        nr = std::min(nc - j, RR);
        for (i = 0; i < mc; i++) {
            for (jj = 0; jj < nr; jj++) {
                Mc[k] = M[(j + jj) * ldM + (i)];
                k++;
            }
            k += (RR - nr);
        }
    }
}

void gemm_blis_generic(int32_t m, int32_t n, int32_t k, float alpha, float* A, int32_t lda, float* B, int32_t ldb, float beta, float* C, int32_t ldc, float* Ac, float* Bc, int32_t mc, int32_t nc, int32_t kc, int32_t mr, int32_t nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf) {

    typedef void (*pack_CB_fn)(float*, float*, size_t, size_t, size_t);
    typedef void (*pack_RB_fn)(float*, float*, size_t, size_t, size_t);
    const pack_CB_fn pack_CB_direct = pack_CB_v_if_AXPY_FP32_Pack_fp32_NR7;
    const pack_RB_fn pack_RB_direct = pack_RB_v_if_AXPY_FP32_Pack_fp32_MR32;


    for (int32_t jc = 0; jc < n; jc += nc) {
        int32_t nc_remain = n - jc;
        int32_t cur_nc = std::min(nc_remain, nc);
        for (int32_t pc = 0; pc < k; pc += kc) {
            int32_t kc_remain = k - pc;
            int32_t cur_kc = std::min(kc_remain, kc);
            int32_t offset_B = jc * ldb + pc;
            float* b_ptr = GET_PTR(B, offset_B);
            int32_t kc_pad = cur_kc;

            int32_t mc_pad = (mc + mr - 1) / mr * mr;
            int32_t nc_pad = (nc + nr - 1) / nr * nr;

            if (cur_nc % nr != 0) {
                int32_t full_nc_elements = (cur_nc / nr) * nr * kc_pad;
                int32_t pad_nc_elements = nr * kc_pad;
                memset(Bc + full_nc_elements, 0, pad_nc_elements * sizeof(float));
            }

            // Counter reads sit outside the chrono window so wall-clock pack time
            // is unperturbed. Both wrappers compile away under -DRVV_INSTRUMENTED=0.
            {
                rvv_perf_region _pack_counters(perf, perf ? &perf->pack : nullptr);
                rvv_region_timer _pack_clock(pack_time);
                if (use_scalar_pack) {
                    pack_CB_scalar(cur_kc, cur_nc, b_ptr, ldb, Bc, nr);
                } else {
                    pack_CB_direct(b_ptr, Bc, cur_kc, cur_nc, ldb);
                }
            }
            *pack_bytes += cur_kc * cur_nc * sizeof(float);

            float beta_actual = GET_BETA(pc, beta);
            for (int32_t ic = 0; ic < m; ic += mc) {
                int32_t mc_remain = m - ic;
                int32_t cur_mc = std::min(mc_remain, mc);
                int32_t offset_A = pc * lda + ic;
                float* a_ptr = GET_PTR(A, offset_A);

                if (cur_mc % mr != 0) {
                    int32_t full_mc_elements = (cur_mc / mr) * mr * kc_pad;
                    int32_t pad_mc_elements = mr * kc_pad;
                    memset(Ac + full_mc_elements, 0, pad_mc_elements * sizeof(float));
                }

                // Counter reads sit outside the chrono window so wall-clock pack time
                // is unperturbed. Both wrappers compile away under -DRVV_INSTRUMENTED=0.
                {
                    rvv_perf_region _pack_counters(perf, perf ? &perf->pack : nullptr);
                    rvv_region_timer _pack_clock(pack_time);
                    if (use_scalar_pack) {
                        pack_RB_scalar(cur_mc, cur_kc, a_ptr, lda, Ac, mr);
                    } else {
                        pack_RB_direct(a_ptr, Ac, cur_mc, cur_kc, lda);
                    }
                }
                *pack_bytes += cur_mc * cur_kc * sizeof(float);

                // Compute counters cover the whole microkernel loop nest, not per tile,
                // to keep read() syscalls off the hot path. In an instrumented build they
                // therefore include the per-tile chrono::now() pair (~200 instructions on
                // jbpi) that Compute_Time_ms excludes; under -DRVV_INSTRUMENTED=0 neither
                // the counters nor that pair exist at all (issue #14).
                {
                rvv_perf_region _compute_counters(perf, perf ? &perf->compute : nullptr);
                for (int32_t jr = 0; jr < cur_nc; jr += nr) {
                    int32_t nr_remain = cur_nc - jr;
                    int32_t cur_nr = std::min(nr_remain, nr);
                    for (int32_t ir = 0; ir < cur_mc; ir += mr) {
                        int32_t mr_remain = cur_mc - ir;
                        int32_t cur_mr = std::min(mr_remain, mr);
                        int32_t row_C = ic + ir;
                        int32_t col_C = jc + jr;
                        int32_t offset_C = col_C * ldc + row_C;
                        float* c_ptr = GET_PTR(C, offset_C);
                        float* a_packed = GET_PTR(Ac, ir * kc_pad);
                        float* b_packed = GET_PTR(Bc, jr * kc_pad);

                        // ~125-160 ns per tile in an instrumented build, which is why
                        // this pair is the thing issue #14 compiles out.
                        rvv_region_timer _kernel_clock(kernel_time);
                        if (cur_mr < mr || cur_nr < nr) {
                            // Fringe tile: use a temp buffer to avoid OOB write by the 32x7 kernel
                            float tmp_C[32 * 7];
                            memset(tmp_C, 0, sizeof(float) * 32 * 7);
                            if (beta_actual != (float)0) {
                                for (int _ii = 0; _ii < cur_mr; _ii++)
                                    for (int _jj = 0; _jj < cur_nr; _jj++)
                                        tmp_C[_jj * 32 + _ii] = c_ptr[_jj * ldc + _ii];
                            }
                            gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, kc_pad, alpha, a_packed, b_packed, beta_actual, tmp_C, 32, mr, nr);
                            for (int _ii = 0; _ii < cur_mr; _ii++)
                                for (int _jj = 0; _jj < cur_nr; _jj++)
                                    c_ptr[_jj * ldc + _ii] = tmp_C[_jj * 32 + _ii];
                        } else {
                            gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, kc_pad, alpha, a_packed, b_packed, beta_actual, c_ptr, ldc, mr, nr);
                        }
                    }
                }
                }
            }
        }
    }
}

}
