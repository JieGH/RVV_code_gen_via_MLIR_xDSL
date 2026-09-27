# Issue #38: B4 shape costs: cheaper A pack and lda/ldc padding off powers of two

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/38

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 4.
**Status:** ⏸️ Issue only. **Implementation on hold.**

## Question
Can B4's shape costs found in #34 be reduced?

## What is known (#34 §5-§6)
- **Small N (~10%)**: `nc` = N = 384, so each packed A block serves 55 tiles (about 145 for the squares). B4 packs 2.3× more elements per flop; packing is 14.7% of its time (7-8% for the squares). A is already packed only once.
- **M = 4096 is a power of two (2-5%)**: C-tile columns 16 KiB apart land in the same L1 sets.
- The reference packs the same way, so it should pay the same cost.

## Candidate levers (not started)
1. A cheaper A pack (it runs at ~0.3 GB/s, IPC 0.03-0.04, instrumented build).
2. Pad `lda`/`ldc` off powers of two in the driver.
3. Check whether other BERT shapes (B1, B5: also N = 384) gain from the same changes.
