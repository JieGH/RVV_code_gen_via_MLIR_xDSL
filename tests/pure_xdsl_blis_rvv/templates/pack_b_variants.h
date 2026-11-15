// Issues #54/#55: B packers that differ only in how the reads are issued.
//
// Every variant writes exactly the Bc of pack_CB_scalar(kc, nc, B, ldb, Bc, RR): B column j
// is at B + j*ldb with k contiguous, panel p covers columns p*RR .. p*RR+nr-1
// (nr = min(nc - p*RR, RR)), element (i, jj) sits at Bc[p*RR*kc + i*RR + jj], and pad columns
// jj >= nr are NOT written. Only the order and the width of the reads change:
//
//   1 row-intr   per panel, per k, per chunk of 8 columns: one strided load (stride ldb*4 B)
//                gathers one element from each of up to 8 columns, one unit-stride store.
//                Same order as the shipped xDSL vector packer, written as C intrinsics.
//                Reads 4 B per column per touch, 8 columns per instruction.
//   2 col-g1     per panel, per column, per strip of up to 8 k (e32m1): unit-stride load of
//                32 B down the column, strided store (stride RR*4 B) into Bc. 1 column at once.
//   3 col-g2     as col-g1, but 2 columns at once: both loads of a strip are issued before
//                both stores.
//   4 col-g4     as col-g1, 4 columns at once.
//   5 col-gall   all columns of the panel at once, capped at 24 (24 live f32m1 registers);
//                a wider panel goes in groups of 24.
//   6 col-g1-m4  one column at a time, strips of up to 32 k (e32m4): 128 B (two lines) per touch.
//
// Variant 0 (`row`) is the shipped xDSL packer; this header does not implement it and
// pack_CB_variant aborts on 0.
//
// On RISC-V the variants use RVV intrinsics (__riscv_ prefixed names, GCC 13). On any other
// target (host tests) a plain C fallback with the SAME loops runs: the same panel / group /
// strip loops, the same bounds helpers, and the same order of "loads" into a small local
// buffer then "stores" out of it. Only the two tiny leaf functions (pack_b_row_chunk,
// pack_b_group_strip<G>, pack_b_col_strip_m4) differ between the branches.
//
// Sizeless RVV vector types cannot live in arrays, and a runtime group size would add
// branches to every k-strip. So the group size G is a template parameter: the intrinsic leaf
// is a recursive always_inline template that loads column N into a local vector, recurses to
// N+1, then stores column N. That issues all G loads before any store (the stores come out in
// reverse column order, which does not change Bc) with no runtime branch. The loops pick the
// instance with ONE switch per group of columns (outside the k-strip loop).
// Strides passed to the intrinsics are in bytes.
#ifndef RVV_PACK_B_VARIANTS_H
#define RVV_PACK_B_VARIANTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv)
#include <riscv_vector.h>
#endif

#define PACK_B_MAX_GROUP 24
#define PACK_B_STRIP_M1 8
#define PACK_B_STRIP_M4 32

enum {
    PACK_B_ROW = 0,
    PACK_B_ROW_INTR = 1,
    PACK_B_COL_G1 = 2,
    PACK_B_COL_G2 = 3,
    PACK_B_COL_G4 = 4,
    PACK_B_COL_GALL = 5,
    PACK_B_COL_G1_M4 = 6,
    PACK_B_NUM_VARIANTS = 7
};

static inline const char* pack_b_variant_name(int v) {
    switch (v) {
        case PACK_B_ROW: return "row";
        case PACK_B_ROW_INTR: return "row-intr";
        case PACK_B_COL_G1: return "col-g1";
        case PACK_B_COL_G2: return "col-g2";
        case PACK_B_COL_G4: return "col-g4";
        case PACK_B_COL_GALL: return "col-gall";
        case PACK_B_COL_G1_M4: return "col-g1-m4";
        default: return "unknown";
    }
}

static inline int pack_b_variant_from_name(const char* name) {
    if (name == NULL) return -1;
    for (int v = 0; v < PACK_B_NUM_VARIANTS; ++v)
        if (strcmp(name, pack_b_variant_name(v)) == 0) return v;
    return -1;
}

// ---- shared bounds helpers (used by both branches) ----
static inline int32_t pack_b_min(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t pack_b_panel_nr(int32_t nc, int32_t j, int32_t RR) { return pack_b_min(nc - j, RR); }

// ---- leaf functions: the only code that differs between intrinsics and fallback ----

#if defined(__riscv)

// One chunk of `cols` (<= 8) columns of row i: strided load across columns, unit-stride store.
static inline void pack_b_row_chunk(const float* src, int32_t ldb, float* dst, int32_t cols) {
    size_t vl = __riscv_vsetvl_e32m1((size_t) cols);
    vfloat32m1_t v = __riscv_vlse32_v_f32m1(src, (ptrdiff_t) ldb * 4, vl);
    __riscv_vse32_v_f32m1(dst, v, vl);
}

template <int N, int G>
static inline __attribute__((always_inline)) void pack_b_group_rec(const float* src, int32_t ldb, float* dst,
                                                                    ptrdiff_t sb, size_t vl) {
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + (ptrdiff_t) N * ldb, vl);
    if constexpr (N + 1 < G) pack_b_group_rec<N + 1, G>(src, ldb, dst, sb, vl);
    __riscv_vsse32_v_f32m1(dst + N, sb, v, vl);
}

// G columns of a k-strip of `len` (<= 8): all G loads, then the G strided stores.
// src = B + j*ldb + i (column 0 of the group), dst = Bc_panel + i*RR + jg.
template <int G>
static inline __attribute__((always_inline)) void pack_b_group_strip(const float* src, int32_t ldb, float* dst,
                                                                      int32_t RR, int32_t len) {
    size_t vl = __riscv_vsetvl_e32m1((size_t) len);
    pack_b_group_rec<0, G>(src, ldb, dst, (ptrdiff_t) RR * 4, vl);
}

// One column, strip of `len` (<= 32) k, e32m4.
static inline void pack_b_col_strip_m4(const float* src, float* dst, int32_t RR, int32_t len) {
    size_t vl = __riscv_vsetvl_e32m4((size_t) len);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(src, vl);
    __riscv_vsse32_v_f32m4(dst, (ptrdiff_t) RR * 4, v, vl);
}

#else  // plain C fallback, same loads-then-stores structure through a local buffer

static inline void pack_b_row_chunk(const float* src, int32_t ldb, float* dst, int32_t cols) {
    float buf[PACK_B_STRIP_M1];
    for (int32_t e = 0; e < cols; ++e) buf[e] = src[(ptrdiff_t) e * ldb];  // vlse32
    for (int32_t e = 0; e < cols; ++e) dst[e] = buf[e];                    // vse32
}

template <int G>
static inline void pack_b_group_strip(const float* src, int32_t ldb, float* dst, int32_t RR, int32_t len) {
    float buf[G][PACK_B_STRIP_M1];
    for (int32_t c = 0; c < G; ++c)  // all G unit-stride loads first
        for (int32_t e = 0; e < len; ++e) buf[c][e] = src[(ptrdiff_t) c * ldb + e];
    for (int32_t c = 0; c < G; ++c)  // then the G strided stores (stride RR)
        for (int32_t e = 0; e < len; ++e) dst[(ptrdiff_t) e * RR + c] = buf[c][e];
}

static inline void pack_b_col_strip_m4(const float* src, float* dst, int32_t RR, int32_t len) {
    float buf[PACK_B_STRIP_M4];
    for (int32_t e = 0; e < len; ++e) buf[e] = src[e];
    for (int32_t e = 0; e < len; ++e) dst[(ptrdiff_t) e * RR] = buf[e];
}

#endif  // __riscv

// ---- loops (identical on both targets) ----

static inline void pack_b_row_intr(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    for (int32_t j = 0; j < nc; j += RR) {
        const int32_t nr = pack_b_panel_nr(nc, j, RR);
        float* Bp = Bc + (ptrdiff_t) j * kc;  // panel p = j/RR starts at p*RR*kc
        for (int32_t i = 0; i < kc; ++i)
            for (int32_t c = 0; c < nr; c += PACK_B_STRIP_M1)
                pack_b_row_chunk(B + (ptrdiff_t) (j + c) * ldb + i, ldb, Bp + (ptrdiff_t) i * RR + c,
                                 pack_b_min(PACK_B_STRIP_M1, nr - c));
    }
}

// One group of G columns: the whole k loop with fixed G (no branch on g inside).
template <int G>
static inline void pack_b_group_cols(const float* src, int32_t ldb, float* dst, int32_t RR, int32_t kc) {
    for (int32_t i = 0; i < kc; i += PACK_B_STRIP_M1)
        pack_b_group_strip<G>(src + i, ldb, dst + (ptrdiff_t) i * RR, RR, pack_b_min(PACK_B_STRIP_M1, kc - i));
}

// One switch per group picks the instance for g = 1..24.
static inline void pack_b_group_dispatch(int32_t g, const float* src, int32_t ldb, float* dst, int32_t RR,
                                         int32_t kc) {
    switch (g) {
#define PBG_CASE(n) case n: pack_b_group_cols<n>(src, ldb, dst, RR, kc); break;
        PBG_CASE(1) PBG_CASE(2) PBG_CASE(3) PBG_CASE(4) PBG_CASE(5) PBG_CASE(6) PBG_CASE(7) PBG_CASE(8)
        PBG_CASE(9) PBG_CASE(10) PBG_CASE(11) PBG_CASE(12) PBG_CASE(13) PBG_CASE(14) PBG_CASE(15)
        PBG_CASE(16) PBG_CASE(17) PBG_CASE(18) PBG_CASE(19) PBG_CASE(20) PBG_CASE(21) PBG_CASE(22)
        PBG_CASE(23) PBG_CASE(24)
#undef PBG_CASE
        default: abort();
    }
}

// G = columns per group; G <= 0 means "all columns of the panel, at most 24 at once".
// The last group of a panel may be narrower (g = nr - jg).
static inline void pack_b_col_groups(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb,
                                     int32_t RR, int32_t G) {
    const int32_t gmax = G > 0 ? pack_b_min(G, PACK_B_MAX_GROUP) : PACK_B_MAX_GROUP;
    for (int32_t j = 0; j < nc; j += RR) {
        const int32_t nr = pack_b_panel_nr(nc, j, RR);
        float* Bp = Bc + (ptrdiff_t) j * kc;
        for (int32_t jg = 0; jg < nr; jg += gmax)
            pack_b_group_dispatch(pack_b_min(gmax, nr - jg), B + (ptrdiff_t) (j + jg) * ldb, ldb, Bp + jg, RR, kc);
    }
}

static inline void pack_b_col_g1_m4(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    for (int32_t j = 0; j < nc; j += RR) {
        const int32_t nr = pack_b_panel_nr(nc, j, RR);
        float* Bp = Bc + (ptrdiff_t) j * kc;
        for (int32_t jj = 0; jj < nr; ++jj)
            for (int32_t i = 0; i < kc; i += PACK_B_STRIP_M4)
                pack_b_col_strip_m4(B + (ptrdiff_t) (j + jj) * ldb + i, Bp + (ptrdiff_t) i * RR + jj, RR,
                                    pack_b_min(PACK_B_STRIP_M4, kc - i));
    }
}

static inline void pack_b_col_g1(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    pack_b_col_groups(B, Bc, kc, nc, ldb, RR, 1);
}
static inline void pack_b_col_g2(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    pack_b_col_groups(B, Bc, kc, nc, ldb, RR, 2);
}
static inline void pack_b_col_g4(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    pack_b_col_groups(B, Bc, kc, nc, ldb, RR, 4);
}
static inline void pack_b_col_gall(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    pack_b_col_groups(B, Bc, kc, nc, ldb, RR, 0);
}

static inline void pack_CB_variant(int variant, const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb,
                                   int32_t RR) {
    switch (variant) {
        case PACK_B_ROW_INTR: pack_b_row_intr(B, Bc, kc, nc, ldb, RR); return;
        case PACK_B_COL_G1: pack_b_col_g1(B, Bc, kc, nc, ldb, RR); return;
        case PACK_B_COL_G2: pack_b_col_g2(B, Bc, kc, nc, ldb, RR); return;
        case PACK_B_COL_G4: pack_b_col_g4(B, Bc, kc, nc, ldb, RR); return;
        case PACK_B_COL_GALL: pack_b_col_gall(B, Bc, kc, nc, ldb, RR); return;
        case PACK_B_COL_G1_M4: pack_b_col_g1_m4(B, Bc, kc, nc, ldb, RR); return;
        default:
            fprintf(stderr, "pack_CB_variant: variant %d (%s) is not implemented in pack_b_variants.h\n", variant,
                    pack_b_variant_name(variant));
            abort();
    }
}

#endif  // RVV_PACK_B_VARIANTS_H
