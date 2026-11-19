# Mixed precision (FP16 A and B, FP32 C): functional check and microkernel C files for Héctor

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/112

**Role / Label:** `needs-triage`. Opened 2026-11-19 at the user's request, after a talk with Héctor about the YOLO collaboration.
**Status:** open. Understanding stage: no code yet. Decisions in §7 come first.
**Branch:** `mixed-precision-fp16-ab-fp32-c-microkernels-for-hector`, cut from `pure-xdsl`.
**Related:** #58 (parent, Héctor's YOLO request), #62 (his GEMM on our board).

---

## 1. What the user asked (2026-11-19)

1. **Correctness first.** Make the mixed-precision design correct: A and B in FP16, C in FP32. Check the current FP32 design stays correct.
2. **Packing:** keep the packing we have. Data is read as FP16. Start with **scalar packing** (easier).
3. **Hand-over files for Héctor:** C source with the microkernels, in two sets:
   - FP32 (A, B, C all FP32).
   - Mixed (A, B FP16; C FP32).
   - Each set: `mr` = 8, 16, 24, 32 and `nr` = 1 … 16 → 64 kernels per set, 128 in total.
4. **His calling convention.** The hand-over kernels must take the same arguments as his. Our own design keeps its own calling method and arguments; only the hand-over files follow his.

## 2. What exists today

### Ours (`tests/pure_xdsl_blis_rvv/`)

- Datatypes: `fp32` and `i8i8i32` (int8 A and B, int32 C). **No FP16, no FP16→FP32.**
- Microkernel (generator `build_gemm`, `tests/test_codeGeneration.py:535`):
  `void gemm_microkernel_v_fp32_<mr>x<nr>_b0|b1(size_t kc, float* alpha, float* A, size_t lda, float* B, size_t ldb, float* beta, float* C, size_t ldc)`
  - Two kernels per tile: `b0` (C = A·B) and `b1` (C += A·B).
  - C registers are `vfloat32m1` (8 floats at VLEN 256); `mr/8` registers per C column.
- `i8i8i32` already widens (`vwmacc`): the pattern to copy for FP16→FP32.
- Scalar packer `render_scalar_pack_functions` is a plain typed copy: works for any element type.
- RVV dialect (`src/xdsltemplate/dialects/rvv.py`) has no FP16 vector types and no `vfwmacc`.

### Héctor's (`martineh/GEMMLib_yolo` @ `b292c92`, `src/GEMM/uk.c`, `gemm_blis.h`)

- Signature, both precisions:
  ```c
  void ukernel_<MR>x<NR>_fp32(int mr, int nr, int kc,
      const float  *Ar, const float  *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
  void ukernel_<MR>x<NR>_fp16(int mr, int nr, int kc,
      const fp16_t *Ar, const fp16_t *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
  typedef _Float16 fp16_t;
  ```
- **His `_fp16` kernels are already mixed precision:** FP16 A and B, FP32 C (`vle16` f16 A, `vfwmacc_vf_f32m2` with an FP16 B scalar). The `NQ_FP16` table in his `Makefile.inc` says "Micro:FP16", but the K1 code accumulates in FP32.
- His `NQ_FP16` mode: input FP32, **pack converts to FP16**, kernel FP16×FP16→FP32, output FP32.
- C is column-major: `Cr[j*Clda + i]`.
- `beta == 0` → C starts at zero; any other `beta` → load C and add. The beta value itself is never used to scale. No `alpha`.
- `mr`, `nr`, `ldA`, `ldB` are not used inside the kernel; it always computes the full MR×NR tile. Edges: his caller runs the kernel into `Ctmp` (`beta = 0`, `Clda = MR`) and copies the valid part.
- Build flags: `-march=rv64gcv_zfh_zvfh` (K1 has Zvfh).
- His existing tiles: 16x16, 16x8, 16x4, 16x1, 64x1, 32x8, 8x8, 8x16 (FP32); 16x8, 16x4, 32x8, 32x4 (FP16).

## 3. Mapping ours onto his signature

**Rule (user, 2026-11-19):** the hand-over files are for Héctor to test **our** microkernels. Only the **arguments** follow his design, so his GEMM can call them in place of his own; the **names** stay ours and do not look like his. Our microkernels keep our own design (registers, LMUL, loop): no copying of his kernel layout.

Names: `xdsl_ukernel_<MR>x<NR>_fp32` and `xdsl_ukernel_<MR>x<NR>_f16f16f32`. They never clash with his `ukernel_*`, so both link together and he points his `uk_selector` at ours.

A thin wrapper per tile, written in the hand-over file:

```c
void xdsl_ukernel_24x8_f16f16f32(int mr, int nr, int kc, const fp16_t *Ar, const fp16_t *Br,
                       float *Cr, float beta, int ldA, int ldB, int Clda) {
    float one = 1.0f;
    if (beta == 0.0f) gemm_microkernel_v_f16f16f32_24x8_b0(kc, &one, Ar, 24, Br, 8, &one, Cr, Clda);
    else              gemm_microkernel_v_f16f16f32_24x8_b1(kc, &one, Ar, 24, Br, 8, &one, Cr, Clda);
}
```

- Our `b0`/`b1` split matches his "beta is zero or not".
- Our generated kernel stays as it is; only the wrapper takes his arguments.
- Packed layout: A panel `MR` per k-step, B panel `NR` per k-step: the same as his (`bA += MR`, `bB += NR`). To be confirmed in step 2 of §5.

## 4. New kernel: FP16 A, FP16 B, FP32 C

Per k-step, per C column `n`, per register `m`:
`C[n][m] = vfwmacc_vf_f32m1(C[n][m], B_f16[n], vle16_v_f16mf2(A + 8m), vl)`

- `vfloat16mf2` (8 halves) widens into `vfloat32m1` (8 floats): same C registers as our FP32 kernel, so the register layout does not change.
- Needs in the RVV dialect: `vfloat16mf2` type, `vle16_v_f16mf2`, `vfwmacc_vf_f32m1`; in the EmitC lowering: `f16` → `_Float16`.
- New generator datatype, for example `f16f16f32` (name to agree, §7).
- Reference for correctness: an FP16×FP16 product is exact in FP32, so a scalar FP32 reference on the same FP16 inputs differs only by summation order. The tolerance can stay tight.

## 5. Plan (in order)

1. **FP32 check, unchanged design.** Run the existing correctness config on the board for `mr` ∈ {8, 16, 24, 32}, `nr` ∈ 1…16. Confirms the baseline before any change.
2. **FP32 hand-over file.** Emit `ukernels_fp32.c` + `.h` with his signature (64 kernels). Test it on the board with a small C driver: each kernel against a scalar reference, `beta` 0 and 1, `Clda` > MR.
3. **Mixed kernel in the generator** (§4). Microkernel test on the board, all 64 tiles, `beta` 0 and 1.
4. **Full GEMM, mixed, scalar packing.** Our GEMM with FP16 A and B in memory, scalar packer copying FP16, FP32 C. Correctness configs on the board (same shapes as `config_correctness.json` plus the YOLO shapes).
5. **Mixed hand-over file.** `ukernels_fp16.c` + `.h`, his signature, tested as in step 2.
6. **Optional, if the user wants it:** build his GEMMLib with `NQ_FP16` and our file in place of his kernels; run his `TEST=T` check. The strongest proof the signature fits.
7. Vector packing for FP16: later, not in this issue.

All C runs on the board (jbpi / jbpi2), core 4, after the usual idle-board check. Correctness only: no timing runs in this issue.

## 6. Risks already seen

- **Register pressure.** M1 accumulators: `mr/8 × nr` C registers + `mr/8` A registers. For 32×8 that is 32 + 4 > 32 vector registers; 32×16 needs 68. Large tiles will spill. Still correct, but slow. Héctor's own 32x8 uses `f32m4` (8 C registers): a different layout. Correctness is the aim here; flag it to Héctor.
- **Name clash:** avoided by the `xdsl_` names (§3).
- **His `NQ_FP16` packs FP32 input to FP16.** The user's plan reads FP16 input. For the hand-over files this does not matter (kernels only); for our full-GEMM test it does.

## 7. Decisions for the user

1. ~~Function names~~ **Decided (user, 2026-11-19):** our own names (`xdsl_ukernel_…`), his arguments, our kernel design (§3).
2. **File split:** one `.c` + `.h` per precision (2 pairs), or one file per kernel (128)?
3. **Input type for our full-GEMM test:** A and B already FP16 in memory (the user's words), or FP32 converted to FP16 during packing (his `NQ_FP16`)?
4. **Datatype name in our generator:** `f16f16f32` (matches `i8i8i32`)?
5. **Step 6** (build his GEMMLib with our kernels): in or out?

## 8. Checklist

- [ ] §7 decisions
- [ ] Step 1: FP32 baseline correctness
- [ ] Step 2: FP32 hand-over file + test
- [ ] Step 3: mixed microkernel + test
- [ ] Step 4: mixed full GEMM, scalar packing
- [ ] Step 5: mixed hand-over file + test
- [ ] Step 6 (if in)
- [ ] Files handed to Héctor
