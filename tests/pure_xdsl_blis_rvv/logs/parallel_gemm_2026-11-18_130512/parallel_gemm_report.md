# Our parallel GEMM on the board: `parallel_gemm_2026-11-18_130512`

```
Issue #56: our GEMM (tile 24x8) at 1 and 8 threads on datasets/yolo26_k1.dat (50 shapes, every row checked)
  Settings : uninstrumented, reference protocol, --tmin 2, --no-nc-overrides, scalar packer
  Cores    : 1 thread on core 4; 8 threads on cores 0-7, OMP_PROC_BIND=close OMP_PLACES=cores (core 0: user's exception, #56 §5)
  Order    : 8 (threads per launch), one board: the first idle one of jbpi
  Load     : the board's processes are read before every launch; busy -> wait 10 min and read again
[13:05] board: jbpi
```

Launches ok: 0 at 1 thread (core 4), 1 at 8 threads (cores 0-7, core 0 included: user's exception, #56 §5). Héctor's numbers: `hector_gemm_2026-11-16_111457` (#62).

WARNING: 0 launch(es) at 1 thread(s), fewer than 3 (house rule 2).

WARNING: 1 launch(es) at 8 thread(s), fewer than 3 (house rule 2).

## Run conditions

- L1 8T: 50 PASS, 0 FAIL, 0 other

- run_2026-11-18_130928: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=2 s), core 0-7, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR24NR8: cpu0 99% cpu1 66% cpu2 65% cpu3 65% cpu4 65% cpu5 65% cpu6 65% cpu7 65%

No shape has launches at both thread counts yet.
