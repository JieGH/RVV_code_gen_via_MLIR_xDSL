# Do the #62 runs match Héctor's PDF? (2026-11-16)

Source: `HectorYolo/Re: Optimizaciones Yolo sobre RISC-V.pdf` (Héctor's email of 13 Nov 2026, forwarding his 10 Nov summary). Its two charts are extracted here:

- `hector_pdf_chart_inference_by_op.png`: inference time per image for 100 images, by optimization step, with persons detected.
- `hector_pdf_chart_phases.png`: the same 100 images in ms, for the base and the first three phases.

**Answer: they cannot be compared directly, and nothing in them conflicts.**

## What each measures

| | Héctor's PDF | our #62 runs |
|---|---|---|
| What is timed | whole-image YOLO26 inference in PyTorch (GEMM + im2col + SiLU + the rest of the network) | his GEMM only (`gemm.bin` → `gemm_rvv`), each of the 53 shapes timed alone, warm buffers |
| Threads | multi-threaded ("es todo multiprocesador") | 1 thread (core 4) and 8 threads (cores 0-7) |
| Board | OrangePi with K1, 3 GB RAM | BananaPi K1: jbpi, jbpi2 |
| Unit | seconds (or ms) per image | GOPs per shape; ms for one pass over the 53 shapes |

## His numbers (read from the charts)

| his version | time per image |
|---|---|
| OpenBLAS | ≈ 1.9 s |
| OP1: his parallel GEMMLib (16x8 / 16x4, GEMV for n = 1, loop-5 split) | ≈ 1.0-1.05 s |
| OP2: + vectorized SiLU | ≈ 0.8 s |
| OP3: + im2col with packing | ≈ 0.7 s |
| OP5: + fp16 GEMM | ≈ 0.6 s |

## Ours (from `../README.md`)

One pass over the 53 shapes at 8 threads: **136.0 ms on jbpi**, **126.2 ms on jbpi2** (21.87 and 23.56 GOPs). At 1 thread: 583.5 and 565.2 ms.

## Why they cannot be matched

1. **Scope.** His times include everything in an inference: PyTorch, im2col, SiLU, layers that are not GEMMs, and data movement. Ours times GEMM calls only.
2. **Call counts.** We do not know how many times each shape is called per image. #61 already lists asking Héctor for these.
3. **Board.** OrangePi against BananaPi: the same K1 chip, but memory and cooling may differ.
4. **No GEMM rate in the PDF.** It gives no GOPs and no `gemm.bin` output.

## What does fit

- If each shape runs about once per image, our 0.13 s of 8-thread GEMM sits well inside his ≈ 1.0 s for OP1.
- He writes that on one thread his GEMM gains almost nothing over OpenBLAS in the real case. That is a statement about OpenBLAS, which we did not run, so our runs neither confirm nor contradict it.

## What a real match needs

From Héctor's OrangePi:
- his own `./gemm.bin gemm_study/yolo.dat output/yolo.csv` output with `TH=8` (and `TH=1`);
- the number of calls per shape in one inference.

Then compare GOPs shape by shape with `../hector_gemm_2026-11-16_*/hector_gemm_report.md`.
