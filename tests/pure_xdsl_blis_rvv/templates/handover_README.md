# xDSL microkernels for GEMMLib_yolo (issue #112)

These are our xDSL-generated RVV microkernels. Each one sits behind a wrapper that takes the same arguments as your `uk_intrinsic_fp32` / `uk_intrinsic_fp16` (GEMMLib_yolo `b292c92`, `src/GEMM/gemm_blis.h`), so your GEMM can call them in place of your own kernels.

## Files

| file | what |
|---|---|
| `xdsl_ukernels_fp32.{c,h}` | FP32 A, B, C. `xdsl_ukernel_<MR>x<NR>_fp32` |
| `xdsl_ukernels_f16f16f32.{c,h}` | FP16 A and B, FP32 C and FP32 sums (like your `_fp16` kernels on K1). `xdsl_ukernel_<MR>x<NR>_f16f16f32` |
| `ref_ukernels.{c,h}` | naive reference: plain loops, no vector code, the same arguments |
| `test_xdsl_ukernels.c` | checks every kernel against the reference |
| `Makefile` | builds and runs the check |

Tiles: MR = 8, 16, 24, 32 and NR = 1 … 16, so each precision has 64 kernels.

## Arguments (the same as yours)

```c
void xdsl_ukernel_24x8_fp32     (int mr, int nr, int kc, const float    *Ar, const float    *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x8_f16f16f32(int mr, int nr, int kc, const _Float16 *Ar, const _Float16 *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
```

- `_Float16` is the same type as your `fp16_t`.
- `Ar`: packed A, MR values per k step. `Br`: packed B, NR values per k step. `Cr`: column-major, `Cr[j*Clda + i]`.
- `mr`, `nr`, `ldA`, `ldB` are not used. The full MR × NR tile is always computed, as in your kernels; edges go through your `Ctmp` path.
- `beta == 0`: C = A·B. Any other `beta`: C += A·B (the value of beta is not used). There is no alpha.
- Only the `xdsl_ukernel_*` names are exported. The kernels inside are `static`, so nothing clashes with your `ukernel_*`.

To use one, point your selector at it. For example, in `uk_selector`, `*uk = xdsl_ukernel_24x8_fp32;` and build with `MR=24`, `NR=8`.

## Build and check

On a K1 board (RVV 1.0, VLEN 256, Zvfh):

```
make run            # gcc-14 -O3 -march=rv64gcv_zfh_zvfh
```

Use gcc-14. gcc 13 does not declare the FP16 widening intrinsics, so the FP16 file does not compile with it.

The check calls every kernel and the reference through your own `uk_intrinsic_fp32` / `uk_intrinsic_fp16` types, so a wrong signature fails to compile. Each kernel runs with:

- `kc` = 1, 7, 256;
- `beta` = 0, 1, 0.5;
- `Clda` = MR + 3 (the 3 extra rows must stay untouched).

It runs two data passes:

- **exact:** whole numbers −5…5. The results are exact, so they must match the reference bit for bit.
- **random:** values in [0, 1). The relative error ‖C − ref‖ / ‖ref‖ must be ≤ max(1e-5, kc·1e-6).

Expected output:

```
fp32      : 64 kernels, 1152 calls, 0 kernels failed -> PASS
f16f16f32 : 64 kernels, 1152 calls, 0 kernels failed -> PASS
```

## Known limits

- **Correctness only:** these kernels have not been tuned for speed in this hand-over.
- **Register pressure:** our kernels keep C in FP32 m1 registers, so a tile needs MR/8 × NR registers for C plus MR/8 for A. Tiles above 32 registers spill. That is MR = 32 with NR ≥ 8, MR = 24 with NR ≥ 10, and MR = 16 with NR = 16. They stay correct but run slowly.
