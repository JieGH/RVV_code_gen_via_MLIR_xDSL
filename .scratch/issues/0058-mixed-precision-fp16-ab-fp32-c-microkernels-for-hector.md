# Mixed precision (FP16 A and B, FP32 C): functional check and microkernel C files for Héctor

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/112

**Role / Label:** `needs-triage`. Opened 2026-11-19 at the user's request, after a talk with Héctor about the YOLO collaboration.
**Status:** open. Implemented, checked on `jbpi`, and audited by three independent agents (§11). Bundle ready to send to Héctor: `.scratch/reports/issue-112/handover/`, zipped as `.scratch/reports/issue-112/xdsl_ukernels_for_hector.zip`.
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
5. **A naive reference in the hand-over (user, 2026-11-19, added during implementation).** The bundle carries a very simple reference microkernel (plain triple loop, no vector code), with the same arguments, so Héctor can see for himself that our kernels compute the right result. The bundle's test checks every kernel against it.

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
- **His `_fp16` kernels take FP16 A and B and store FP32 C, but only two of them sum in FP32.** `16x8` and `16x4` use `vfwmacc_vf_f32m2` (FP32 sums). `32x8` and `32x4`, the ones his `NQ_FP16` GEMM uses by default (`YOLO_MR` = 32), sum in FP16 (`vfmacc_vf_f16m2`) and widen only at the store. *Corrected 2026-11-19 after the independent audit (§11); the first version said all of them sum in FP32.*
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

1. **FP32 baseline, unchanged design.** Existing `config_correctness.json` (tiles 16x15, 32x6, 32x7, `correctness_small.dat`, oracle on) on the board.
2. **Mixed kernel in the generator** (§4): datatype `f16f16f32`.
3. **Hand-over bundle** (`gen_handover_ukernels.py`): `xdsl_ukernels_fp32.{c,h}`, `xdsl_ukernels_f16f16f32.{c,h}` (64 tiles each), the naive reference `ref_ukernels.{c,h}`, `test_xdsl_ukernels.c`, `Makefile`, `README.md`. Test on the board (§5a).
4. **Full GEMM, mixed, scalar packing.** `f16f16f32` in the pipeline: driver A/B `_Float16`, C `float`; scalar packer only (a run without `--scalar-pack` is refused); new `config_issue112_mixed_correctness.json` (16x15, 32x6, 32x7, 24x8 on `correctness_small.dat`, oracle on).
5. Vector packing for FP16: later, not in this issue.

All C runs on the board (jbpi / jbpi2), core 4, after the usual idle-board check. Correctness only: no timing runs in this issue.

### 5a. The bundle test

- Calls every wrapper through **his** typedefs (`uk_intrinsic_fp32`, `uk_intrinsic_fp16`, copied from `gemm_blis.h`): a signature mismatch does not compile.
- Compares each kernel with the **naive reference** `ref_ukernel_fp32` / `ref_ukernel_f16f16f32` (same arguments; triple loop; FP32 sums).
- `kc` ∈ {1, 7, 256}; `beta` ∈ {0, 1, 0.5} (0.5: his rule, C += A·B, value unused); `Clda` = MR + 3, and the 3 extra rows per column must stay untouched.
- **Exact pass:** whole numbers −5…5; exact in FP16, products and sums exact in FP32: our result must equal the reference bit for bit.
- **Random pass:** values in [0, 1), rounded to FP16 for the mixed kernels; relative norm error ≤ `max(1e-5, kc·1e-6)` (our GEMM driver's rule).
- Built with **gcc-14**: the board's default gcc 13.2 does not declare the FP16 multiply-add intrinsic (`__riscv_vfwmacc_vf_f32m1` with an FP16 scalar), so the mixed file fails to compile there (seen on jbpi, 2026-11-19). Héctor builds with gcc-14 too.

## 6. Risks already seen

- **Register pressure.** M1 accumulators: `mr/8 × nr` C registers + `mr/8` A registers. For 32×8 that is 32 + 4 > 32 vector registers; 32×16 needs 68. Large tiles will spill. Still correct, but slow. Héctor's own 32x8 uses `f32m4` (8 C registers): a different layout. The README warns about it.
- **Name clash:** avoided by the `xdsl_` names (§3); our inner kernels are `static`, so only the wrappers are visible.
- **His `NQ_FP16` packs FP32 input to FP16.** Our full-GEMM test reads FP16 input (§7.3). The hand-over files are kernels only, so this does not affect them.

## 7. Decisions (settled 2026-11-19)

1. **Names:** our own (`xdsl_ukernel_<MR>x<NR>_fp32`, `xdsl_ukernel_<MR>x<NR>_f16f16f32`), his arguments, our kernel design (§3).
2. **Files:** one `.c` + `.h` pair per precision.
3. **Full-GEMM input:** A and B already FP16 in memory.
4. **Datatype name:** `f16f16f32`.
5. **Building his GEMMLib with our kernels:** out (his code stays unchanged; he can run it himself).
6. **Bundle test** on the board, also shipped to him (§5a).
7. **His semantics in the wrapper:** `mr`, `nr`, `ldA`, `ldB` unused; `beta == 0` overwrites, other `beta` adds; no alpha.
8. **FP16 type:** our header uses `_Float16` (the same type as his `fp16_t`); no typedef of ours.
9. **Mixed kernel:** `vfloat16mf2` A, `vfwmacc_vf_f32m1` into our FP32 m1 accumulators, FP16 B scalar.
10. **All 64 tiles shipped**, spilling ones included, with a README warning.
11. **Full GEMM:** scalar packing only; runs as §5 step 1 and 4.
12. **Locations:** generator `tests/pure_xdsl_blis_rvv/gen_handover_ukernels.py`; bundle committed in `.scratch/reports/issue-112/handover/`; the user sends it to Héctor.
13. **Glossary:** *Mixed Precision (f16f16f32)* and *Hand-over Kernel* in `CONTEXT.md`.
14. **Naive reference in the bundle** (§1.5).

## 8. Checklist

- [x] §7 decisions
- [x] Step 1: FP32 baseline correctness on the board (§9)
- [x] Step 2: mixed kernel in the generator
- [x] Step 3: hand-over bundle + naive reference + board test (§9)
- [x] Step 4: mixed full GEMM, scalar packing, board correctness (§9)
- [x] Code review (§10)
- [x] Bundle committed
- [ ] The user sends `.scratch/reports/issue-112/handover/` to Héctor

## 9. Results (jbpi, core 4, correctness only, 2026-11-19)

| check | build | result | log |
|---|---|---|---|
| Hand-over bundle: 64 FP32 + 64 f16f16f32 kernels against the naive reference, 1152 calls each (§5a) | bundle Makefile, gcc-14 14.2, `-O3 -march=rv64gcv_zfh_zvfh` | **PASS**, 0 kernels failed in both sets | `.scratch/reports/issue-112/handover/logs/board_jbpi_2026-11-19_145406.log` |
| Same test with two planted bugs (b1 called for `beta = 0` in fp32 24x8; `Clda + 1` in f16f16f32 16x5), throwaway copy, not committed | as above | **FAIL** on exactly those two kernels (519 and 240 wrong elements): the test can fail | (scratch copy, deleted) |
| Full GEMM FP32, unchanged design: `config_correctness.json` (16x15, 32x6, 32x7 × `correctness_small.dat`) | uninstrumented, reference protocol, `--tmin 0.2`, `--no-nc-overrides`, vector packing, default gcc 13.2 | **9 / 9 PASS** | `tests/pure_xdsl_blis_rvv/logs/run_2026-11-19_143324/` |
| ~~Full GEMM f16f16f32, first run~~ **VOID**: the driver filled the FP16 inputs with NaN and the check let NaN pass (§11.1); its 12 "PASS" rows compared `nan == nan`, and its GOPs were timed on NaN data | — | **void** | `tests/pure_xdsl_blis_rvv/logs/run_2026-11-19_144226/`, `.scratch/reports/issue-112/console/issue112_mixed_correctness.out` (kept as evidence) |
| Full GEMM f16f16f32, rerun with the fixed driver: `config_issue112_mixed_correctness.json` (16x15, 32x6, 32x7, 24x8 × `correctness_small.dat`) | uninstrumented, reference protocol, `--tmin 0.2`, `--no-nc-overrides`, `--scalar-pack`, g++-14 `-march=rv64gcv_zfh_zvfh` (build output) | **12 / 12 PASS**, samples real (e.g. `33.47 == 33.47`) | `tests/pure_xdsl_blis_rvv/logs/run_2026-11-19_155626/`, `.scratch/reports/issue-112/console/issue112_mixed_correctness_v2.out` |
| Full GEMM FP32 rerun with the fixed driver (`config_correctness.json`) | as the first FP32 run | **9 / 9 PASS**, same sample values as before | `tests/pure_xdsl_blis_rvv/logs/run_2026-11-19_160242/`, `.scratch/reports/issue-112/console/issue112_fp32_correctness_v2.out` |
| f16f16f32 binary without `--scalar-pack` | as above | refused: "f16f16f32 runs with scalar packing only …" | — |
| Host tests | — | 651 passed (full suite), then 71 passed after the §10 fixes (`test_handover_ukernels.py`, `test_board_pin.py`) | — |

No timing numbers are quoted from these runs: they are correctness runs (short `tmin`, board not checked idle).

## 10. Code review (2026-11-19, Standards and Spec axes)

Fixed:
- `ref_baseline/generated/` side effects of the host tests restored (`git checkout`).
- Board provenance now also records `g++-14` (`board_pin.sh`), since f16f16f32 projects build with it.
- `run_issue112_handover.sh` uses the shared `pin_check` (core 0 refused unless `--allow-core-0`); its log header says correctness only and names the build.
- f16f16f32 pack loop: `pack_gen_dims = []` up front instead of an inline conditional.
- `mlir-translate` path kept in one place (`gen_handover_ukernels.DEFAULT_TRANSLATE`).

Kept on purpose:
- The bundle Makefile repeats the gcc-14 / Zvfh flags: Héctor gets that folder alone.
- No kernel dataclass in `gen_handover_ukernels.py`: the 3 helpers are small.
- The reference cast to `C_TYPE` in `gemm_ref` applies to every datatype: no change for fp32 and int8.

## 11. Independent audit (2026-11-19, three agents, before the hand-over)

Each agent started without this conversation and checked the committed files, Héctor's tree and the board, not our claims.

### 11.1 Logic audit: one real bug, fixed

- **Bug (full-GEMM driver, FP16 only):** `fill_random` did `static_cast<T>(rand()) / RAND_MAX`. With `T = _Float16`, `rand()` above 65504 and `RAND_MAX` become inf, so most inputs were inf/inf = NaN. The check `error > tol` is false for NaN, so NaN passed. Fixed in `templates/driver_benchmark.cpp`: divide in float, then convert (the same values as before for fp32); the check is now `!(error <= tol)`, so NaN fails for every datatype.
- **Old tests not affected:** fp32 never overflowed (2^31 fits in float); int8 has its own fill. A scan of every log under `tests/pure_xdsl_blis_rvv/logs/` found no NaN or inf sample in any run other than #112's first mixed run (312 files with sample lines; 2442 PASS rows, none with NaN in the time or GOPs columns).
- **Kernels:** all 128 generated kernels (b0 and b1) run through a C-subset interpreter, kc ∈ {1, 5, 16}: 768 cases bit-exact against numpy; vl, offsets, b0 never reading C, the Clda padding rows, the intrinsic argument order (RVV 1.0) all checked. fp32 path, Makefile switch and `gemm_ref` cast: no change for fp32 and int8.
- Blind spots noted (not bugs): the bundle test passes `ldA = ldB = 0`; no NaN initial C in the bundle test (covered by 11.3); all MR are multiples of 8, so no partial-vl chunk; `get_optim_mc_nc_kc(sizeof(float), …)` ignores the 2-byte A/B (blocking only).

### 11.2 Spec audit

- All of §1 and §7 met; 64 + 64 tiles in the code, headers and test table; arguments, packed layout (`pack_generic_MRxkc_A`, `pack_generic_B`), the `Ctmp` edge path and his `betaI` use all match; no symbol clash.
- Fixed: the README pointed to a `uk_selector` that does not exist; it now names `gemm_blis.c:858-863` and `972-977`. §2 and the README wrongly said all his `_fp16` kernels sum in FP32: his default `32x8` / `32x4` FP16 kernels sum in FP16 (checked: `vfmacc_vf_f16m2`, `uk.c:584-656`). Our results will differ from his, closer to FP32; the README says so.

### 11.3 Independent board rerun (own test program, jbpi2, core 4)

Own reference (double), own seed; kc ∈ {0, 1, 2, 3, 64, 513, 2000}; Clda ∈ {MR, MR + 17}; beta ∈ {0, 1, 0.5}; NaN in C before every beta = 0 call; negatives; FP16 values up to ±1000; guard zones around Ar, Br, Cr; Ar and Br unchanged. Files: `.scratch/reports/issue-112/audit/`.

| set | compiler | calls | failed |
|---|---|---|---|
| fp32 | gcc-14 14.2 | 5376 | 0 |
| f16f16f32 | gcc-14 14.2 | 8064 | 0 |
| fp32 | gcc 13.2 | 5376 | 0 |
| f16f16f32 | gcc 13.2 | — | does not compile (`vfloat16mf2_t` unknown), as the README says |

### 11.4 Added after the audit

- `README_es.md`: Spanish copy of the README in the bundle (user, 2026-11-19).
- `.scratch/reports/issue-112/xdsl_ukernels_for_hector.zip`: the bundle as one zip.

