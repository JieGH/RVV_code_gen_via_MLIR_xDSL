# Issue #49: Where the B read streams land in memory: why B3_K256 breaks the "NR up to 6 is steady" rule of #48 (sub-issue of #48)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/49

**Report:** [issue-48-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-48-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-48-brief.tex)), "Next steps" — updated 14 October 2026 with this issue's desk-table result.

**Comes from:** #48 §8.3 and §8.5 (answer 6, "K matters"), and the B3_K256 tables of the #48 report
(`tests/pure_xdsl_blis_rvv/logs/issue48_2026-09-30_report.md`, "jbpi, B3_K256" and "jbpi2, B3_K256").

**Role / Label:** `needs-triage` — sub-issue of #48, priority 1 of the #48 follow-ups (opened 2026-10-14). Step 1 done (§8, below): no single desk quantity splits it. The `--pad-ldb` driver option and `run_issue49.sh` (step 2) are built and host-validated; not yet run against the real boards.
**Status:** 🟡 Open. Step 1 done; step 2 ready, awaiting board access.
**Target:** step 1 none (desk, done); step 2 `jbpi` and `jbpi2`, core 4, house rules of #39.
**Shape(s):** B3 (64x384x384) and B3_K256 (64x384x256); B2 and Reuse24 as steady controls.
**Primary Sources:**
- [[0032-b3-microkernel-dimension-sweep-pack-and-compute]] (#48) — the 25-tile sweep; §7.1 has the packer table, §8.2-8.3 the results.
- [[0025-b3-slow-state-root-cause]] (#41) §8.4 — the slow state is the L1D prefetcher doing about half its work; what makes it back off is open.
- [[0030-reference-b3-per-rep]] (#46) §7.2 — one heap allocation that moved the buffers changed the reference's B3_K256 by about 30%: placement matters.
- [[0028-pack-b-fewer-streams-spec]] (#44) — the "fewer streams" fix, on hold; its premise is the thing this issue tests.
- `tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp:752` — `int lda = m, ldb = k, ldc = m;`: B is column-major with `ldb = K`.
- `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py:556` — `offset_B = jc * ldb + pc`.
- `tests/pure_xdsl_blis_rvv/pack_builder.py` (`build_pack_CB_v_if`, the `vlse32` at line 79/192) — the B packer reads NR columns at once, `ldb`*4 bytes apart.
- `tests/pure_xdsl_blis_rvv/pack_dims_table.py` — the #48 table; the place to add the new columns.

---

## 1. Question

#48 found that on B3 (K=384) every tile with NR up to 6 is steady and every tile with NR of 7 or more is two-speed.
It read this as "the number of B read streams decides it" (#48 §8.5, answer 2), which is the premise of #44.
The B3_K256 rows of the same launches do not follow that rule (`jbpi`, vector packer, slow-rep share):

| tile | B3 (K=384) | B3_K256 |
|---|---|---|
| 16x4, 32x4, 8x4, 4x4 | 0% | 0% |
| 16x6 | 0% | **48%** |
| 24x6 | 0% | **81%** |
| 32x6 | 0% | **19%** |
| 16x8 | 89% | **0%** (6.76 GOPs) |
| 24x8 | 87% | **0%** (7.68 GOPs, the best B3_K256 number) |
| 12x8 | 82% | **0%** |
| 4x8, 8x8 | 82%, 73% | 93%, 95% |
| NR of 12 and up | 53-93% (16x16, over the register budget: 25%) | 76-96% |

At the same NR the state flips with K. The B packer reads column j of a panel at `B + (jc+j)*ldb + pc`, and `ldb = K`,
so the step between the NR read streams is **1536 bytes at K=384 and 1024 bytes (a power of two) at K=256**.
**Is the slow state decided by how many streams there are, or by where the streams (and the pack buffers) land in memory?**

## 2. Why it matters (why priority 1)

- If it is where the streams land, #44 (one stream at a time) may fix B3 for the wrong reason, or not at all at K=256,
  and a cheaper fix (padding `ldb`, moving a buffer) may do the job.
- #46 §7.2 already showed that moving the buffers by one heap allocation changes the reference's B3_K256 by about 30%.
- #41 knows *what* the slow state is (the L1D prefetcher backs off) and not *what makes it back off*. A layout rule would be that answer.

## 3. Things that could decide it (to tell apart)

- **H1, stream count.** NR alone (the #48 reading). Predicts the same answer at K=256 and K=384: already contradicted at NR=6 and NR=8.
- **H2, where the stream heads sit in a page or cache set.** With a step of 1024 B, columns j and j+4 sit at the same offset in a 4 KiB page;
  with 1536 B, columns j and j+8 do (8 x 1536 = 3 x 4096). How many streams share a page offset, and how many 4 KiB pages one k-step
  touches, both change with K and with NR. (The L1D geometry of the X60, its size, ways and line size, is to be read from the manual or the device tree before use, not assumed.)
- **H3, kc and the K blocks.** kc, the number of K blocks and the size of the last one change with the tile and with K (#48 §7.1).
  At K=256, tiles with kc=256 run one K block; at K=384 they run two (256 + 128).
- **H4, where the buffers start.** The driver aligns A, B, C, Ac and Bc to 32 bytes only (#46 §7.2), so their start offsets in a page
  depend on what was allocated before them, and differ between builds.
- **H5, MR together with NR.** At K=256, NR=8 is steady at MR of 12, 16 and 24 and slow at MR of 4 and 8. Some rule that uses both
  (bytes of Ac against bytes of Bc, number of A panels) may be what is left after H2-H4.

## 4. Plan

1. **Desk table, no board.** Extend `pack_dims_table.py` (or a new read-only script) to print, for every #48 tile at K=384 and at K=256:
   the B stream step in bytes; how many distinct offsets mod 4096 (and mod the L1D set span, once known) the NR stream heads take; how many
   4 KiB pages one k-step of one panel touches; kc, the number of K blocks and the last block's size; Bc panel bytes; and the measured
   slow-rep share of both boards from the #48 report. Then look for **one quantity that puts every steady case on one side and every slow
   case on the other, in both K columns**. Write the table and the result into this issue before any board time.
2. **The decisive board test: change `ldb` alone.** Add a driver option that allocates B with a padded leading dimension
   (`ldb = K + pad`, data the same, results still checked against `gemm_ref`), then run at a fixed tile and fixed K:
   - tiles 16x6 and 16x8 (they flip in opposite directions), shapes B3 and B3_K256;
   - pad of 0, 16 floats (64 B), 8 floats (32 B), and one pad that turns K=384's step into a power of two (`ldb = 512`), and one that turns
     K=256's step into K=384's (`ldb = 384`).
   If the slow state follows `ldb` at a fixed NR and kc: H2. If it stays with NR: H1. If it follows kc only: H3.
3. **Buffer start, only if step 2 does not explain it.** A driver option that shifts Bc (then Ac) by 0, 64, 1024 and 2048 bytes from a 4 KiB boundary, on the same tiles. This tests H4.
4. **Repeats and rules.** Uninstrumented, reference protocol, `--tmin 10`, `--no-nc-overrides --cpu 4`, vector packer; 3 launches per board,
   A-B-B-A order, the B2 and Reuse24 controls in the same launches, other users read before and after every run.

## 5. How to read it

- **One layout quantity splits both K columns (step 1) and padding `ldb` moves the state (step 2):** the trigger is where the streams land.
  #44's premise is wrong or only part of it. The next step is a fix by padding or placement, specced separately.
- **Padding does nothing and the state stays with NR:** H1 holds and the K=256 flips need a second cause (H3/H5). #44 stays the fix to test.
- **Only buffer start moves it (step 3):** the fix is alignment of the pack buffers (and #46's reference result has the same cause).

## 6. Out of scope

Changing the shipped tile, packer or alignment (any fix is a separate spec); #43 and #44 (on hold); the reference design.

## 8. Step 1: the desk table (2026-10-14)

`issue49_layout_table.py` extends `pack_dims_table.py`'s model to print, for every #48 tile at
K=384 and K=256: the B stream step in bytes (`ldb * 4`, `ldb = k` today), how many distinct
byte offsets mod 4096 the NR stream heads take, how many 4 KiB pages one k-step of one panel
touches, `kc`/K blocks/last block, Bc panel bytes, `mc` and A panels — joined with the measured
slow-rep share of both boards (vector packer) from `logs/issue48_2026-09-30_report.md`. Full
table: run `python3 issue49_layout_table.py`.

**Result: no single quantity among the candidates tried puts every steady tile on one side and
every slow tile on the other, in both K columns at once.**

- **At K=384 alone**, `pages_per_kstep == 2` (equivalently NR ≤ 6, since the step is 1536 B and
  the period mod 4096 is 8) is exactly the steady/slow line — this is #48's own rule restated,
  not a new one.
- **At K=256**, that same quantity fails: 16x6, 16x7, 8x8, 24x6, 32x6, 32x7 and 4x8 all have
  `pages_per_kstep == 2` but are slow, while 16x8, 24x8 and 12x8 also have `pages_per_kstep == 2`
  and are steady. "Distinct offsets mod 4096" collapses to 4 for every NR ≥ 4 at K=256 (the step,
  1024 B, has period 4), so it cannot split anything there either.
- **H5 (MR together with NR) holds at one NR, not in general.** At K=256, NR=8 is steady at
  MR 12/16/24 and slow at MR 4/8 (exactly as #49 §3 observed) — but NR=6 at K=256 is slow at
  MR=16 regardless, and NR=4 is steady at every MR tried. No MR threshold tried (`mc`, A panel
  count, `mr ≥ 12`) reproduces the NR=8 line at every other NR.
- `kc`/K-blocks/last-block-size (H3) do not split cleanly either: K=384's three-K-block tiles
  (kc=128) include both steady (32x4, 32x6) and slow (16x12..16x16, 32x7) cases.

**Reading this (per §5):** none of H1 (stream count alone), H2 (page/offset geometry alone), H3
(kc/K-blocks alone) or a simple H5 (MR threshold alone) survives as a single desk-computable
rule across both K. This does not clear any hypothesis — it means the desk table cannot tell
H1-H5 apart on its own, which is the case the decisive board test (step 2) exists for. Proceed
to step 2.

## 9. Step 2: the driver option and its host-side check (2026-10-14)

Added `--pad-ldb <elements>` to the driver (`templates/driver_benchmark.cpp`, wired through
`compile_macro_bench.sh`): `ldb = k + pad` instead of `ldb = k`, `k`/the data/the correctness
check unchanged (`pad_ldb = 0` reduces to today's behaviour exactly — logs keep their current
names). B's buffer follows `ldb`, not `k` (`pad_ldb_rows = ldb + 32`), so a padded run cannot
read past its own allocation.

Before spending board time: `test_pad_ldb.py` builds the real `gemm_dispatch_xdsl.cpp` (as
`test_pad_zeroing.py`/`test_zero_pack_lines.py` do) and checks the GEMM is still exactly right
with B allocated as `ldb x n`, `ldb > k`, for pads {0, 8, 16, 128} — the same deltas the issue
asks for (K=384: ldb 384/392/400/512; K=256: ldb 256/264/272/384). All pass: the pack/dispatch
code already tolerates an arbitrary `ldb`, so the board test measures a layout effect, not a
latent correctness bug.

`run_issue49.sh` runs the step-2 series: `config_issue49_sweep.json` (16x6 and 16x8, the two
tiles that flip in opposite directions), one run per pad, pads rotating between launches (house
rule 2), both boards taking turns, the shared lock and RED NOTE conventions of `run_issue46.sh`/
`run_issue48.sh`. Validated with `--dry-run` and against a stubbed board (`test_run_issue49.py`,
22 tests green together with `test_run_issue50.py`) — not yet run against the real boards: this
host's jump host (`fej`) does not resolve from here. Run it from a host that can reach the
boards:

```bash
setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue49.sh --start-at 00:00 \
    > tests/pure_xdsl_blis_rvv/logs/issue49.out 2>&1 < /dev/null &
```

## 9.1. Code review note: "A-B-B-A order" (2026-10-14)

§4.4 says "3 launches per board, A-B-B-A order". That phrase is house rule 2's (`execution.md`)
pattern for alternating **two** settings; this test compares **four** pads at once, for which a
literal two-symbol ABBA has no clean reading. `run_issue49.sh` instead cyclically rotates the
four pads between launches (launch 1: 0 8 16 128; launch 2: 8 16 128 0; launch 3: 16 128 0 8) —
the same multi-setting rotation `run_issue46.sh` (O/P/R) and `run_issue48.sh` (the arms) already
use for more than two settings, which still satisfies the rule's actual intent (no pad is always
first or last). Flagged by the Spec review as a deviation from the issue's literal wording;
recorded here as a deliberate, precedented choice rather than silently resolved.

## 10. Closure

- [x] Desk table for both K, with the result (one splitting quantity, or none), written here before any board run. See §8: none found.
- [x] Driver option for a padded `ldb`, with a test that the result still checks. See §9: `--pad-ldb`, `test_pad_ldb.py`.
- [ ] Step 2 run on both boards, 3 launches, logs committed, no red note (or the red note in the report). `run_issue49.sh` is ready; not yet run (no board access from this session).
- [ ] Written answer: stream count, stream placement, kc or buffer start, and what it means for #44.
