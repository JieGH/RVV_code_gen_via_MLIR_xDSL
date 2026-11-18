# Issue #54: Which property of the B packer's reads sets off B3's slow state? Try small read-order changes on B3 only (follow-up of #48, #49, #53)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/54

**Report:** [issue-54-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-54-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-54-brief.tex)), 23 October 2026.

**Comes from:** #48 §8.5 (the slow state follows NR; the scalar packer is far steadier at NR ≥ 8), #53 §7 (the scalar packer wins by
being steadier, not cheaper: same fast-rep pack time), #49 §11 (padding `ldb` changes the state at a fixed NR, so stream count
alone is not the cause), #50 §8 (a slow rep halves the L1D prefetch hits in both pack and compute).

**Role / Label:** `needs-triage` — opened 2026-10-22 at the user's request. Investigation ticket; the code it needs is spec #55
[[0039-pack-b-read-order-variants-spec]]. Goal: a report in the week of 2026-10-26 with something new about the design.
**Status:** ✅ Closed 2026-11-18 (housekeeping: answered, results merged). Was: 🔬 Series 1 and 2 done (2026-10-22/23, §8.1-8.3). The trigger is property C (4 B per column per touch). Reading one column at a time (`col-g1-m4`) removes B3's slow state on every tile and keeps B2 and Reuse24. Brief next.
**Target:** `jbpi` (and `jbpi2` if no other user's job is on it), core 4, house rules of #39.
**Shape(s):** B3 (64×384×384) under test; B3_K256 (64×384×256) because K moves the rule (#48 §8.3); B2 (384×384×64) and Reuse24 (384³)
as the "do not make it worse" controls. All four are `datasets/issue46_ref_b3.dat`.
**Tiles:** 16x15 (shipped), 16x8 (the worst B3 tile, 89-94% slow, and no fringe tiles at all, so the fringe path of #52 cannot be the cause there),
16x6 (steady on B3: a variant must not break it).
**Primary Sources:**
- [[0032-b3-microkernel-dimension-sweep-pack-and-compute]] (#48) §7.1 (what the vector B packer does per NR), §8.2-8.5.
- [[0037-scalar-vs-vector-packer-on-controls]] (#53) §7; [[0033-b-stream-layout-k256-flip]] (#49) §11; [[0034-prefetcher-counters-four-corners]] (#50) §8.
- [[0028-pack-b-fewer-streams-spec]] (#44) — the one-column read order, specified 2026-09-27, on hold. This ticket takes its idea, in a
  cheaper form (C intrinsics, no dialect work), and widens it to a graded set.
- `tests/pure_xdsl_blis_rvv/pack_builder.py:50-87` (`build_pack_CB_v_if`, the shipped vector B packer: one `vlse32` per k across 8 columns),
  `generate_pure_blis_project.py:213-226` (`pack_CB_scalar`), `:247-275` (`render_timed_pack_block`, the `use_scalar_pack` switch), `:584-591` (the B pack call).

---

## 1. Question

The vector and the scalar B packer read **the same addresses in the same order** (#36, #52 rows 3-4), and their pack times are equal when both
are fast (#53: 16x8, 0.76 ms against 0.77-0.80 ms). Yet on B3 at 16x8 the vector packer is 89-94% slow reps and the scalar one 2-8%
(#48 §8.2). Three things differ between the two, and no run so far has pulled them apart:

| property | shipped vector packer | scalar packer |
|---|---|---|
| **A. How a read is issued** | one strided `vlse32` touches 8 lines (8 columns, 1536 B apart at K=384) | one load per element, one line each |
| **B. How many columns are read at once** | NR read streams side by side | NR read streams side by side |
| **C. Bytes taken from one column per touch** | 4 B | 4 B |

B and C are the same in both, so A is the obvious suspect. But #44's stream-count idea (B) has never been tried, and C (read a column 32 B or
128 B at a time, as a prefetcher likes) has never been tried either.

**Which of A, B and C sets off the slow state on B3, and is there a B packer that keeps B3 out of it without costing B2?**

## 2. Why this direction (and why not a bigger one)

- **Small and local.** Only the B packer changes, behind a switch, off by default. The packed layout of `Bc`, the microkernel, the blocking
  (mc, nc, kc), the A packer and the cache model stay exactly as they are. The user asked for a gradual start on the obvious, not an overhaul.
- **Cheap to build.** The variants are C intrinsics beside the existing scalar packer, which is already chosen at run time
  (`use_scalar_pack`). No xDSL dialect op is needed (the dialect has no fp32 strided store yet, #44). A variant that wins is ported to
  xDSL afterwards, as its own step.
- **Each variant moves one property.** So a result names a property, which is the new insight for the design, whatever the speeds turn out to be.
- **It also asks a question never asked before:** can the B packer fall into the slow state **on its own**, without the compute between packs?
  (#41-#50 only ever saw pack and compute together.) A pack-only loop answers it.

## 3. The variants (spec #55 has the details)

| name | how B is read | A: issue | B: columns at once | C: bytes per column per touch |
|---|---|---|---|---|
| `row` (shipped, xDSL) | `vlse32` across 8 columns per k | vector, strided | NR | 4 |
| `row-intr` | the same, written as C intrinsics | vector, strided | NR | 4 |
| `scalar` (exists, `--scalar-pack`) | one element at a time | scalar | NR | 4 |
| `col-gall` | `vle32` 8 floats down each column, all NR columns side by side, then strided stores into `Bc` | vector, unit-stride | NR | 32 |
| `col-g4` | the same, 4 columns at a time | vector, unit-stride | 4 | 32 |
| `col-g2` | the same, 2 columns at a time | vector, unit-stride | 2 | 32 |
| `col-g1` | one column at a time (#44's order) | vector, unit-stride | 1 | 32 |
| `col-g1-m4` | one column at a time, LMUL=4: 32 floats per load | vector, unit-stride | 1 | 128 (two whole lines) |

`row-intr` is a control: it must behave like `row`. If it does not, the C path itself changes something and every other variant's result
has to be read against `row-intr`, not `row`.

## 4. How to read it

The full-run slow-rep share of B3 at 16x8 and 16x15 is the main reading; the pack-only loop is the second.

- **`row` slow, `col-gall` steady:** property **C or A** (the way each column is read), not the count. `col-gall` keeps NR streams.
  Compare with `scalar` (also NR streams, 4 B, but steady): if both are steady, the trigger is **one instruction touching 8 lines**
  (property A). This is the reading the earlier data points to.
- **`col-gall` slow, and the slow share falls as g goes NR → 4 → 2 → 1:** property **B**, stream count. #44's idea holds after all,
  despite #49 (#49 padded `ldb`, which moves both the count of distinct pages and where the lines land).
- **Every `col-*` slow, `scalar` steady:** neither B nor C. The vector unit's strided or unit-stride reads both trigger it, scalar loads
  do not. Then the lever is the scalar packer for B3, or mixing scalar and vector (a later issue).
- **The pack-only loop goes two-speed for `row`:** the packer alone sets off the state. **It stays one-speed:** the state needs pack and
  compute together (the compute's reads of `Bc` and C after the pack), and the next step is in the order of pack and compute, not in the packer.
- **Any variant that wins B3 but loses B2 or Reuse24** by more than the launch-to-launch spread is not a fix for the shipped build,
  only a per-shape option (as #53 found for the scalar packer).
- **16x6 must stay steady** on B3 under every variant. If a variant breaks it, that variant changes something beyond the read order.

## 5. Out of scope

Changing `Bc`'s layout, the A packer, the microkernel, the tile, the blocking or the cache model; `cbo.zero` (#43); `ldb` padding (#49);
allocating Ac/Bc once (#52 rank 2, a candidate next step if this ticket rules the packer out); the fringe path (#52 rank 1); porting a
winner to xDSL (a separate step after the report); shipping any variant by default.

## 6. Plan and dates (target: report in the week of 2026-10-26)

| when | what | board |
|---|---|---|
| desk, day 1-2 | spec #55 phases 1-4: variants, switch, tests, pack-only loop, series script, report script | none |
| night 1 | series 1: `row`, `row-intr`, `scalar`, `col-gall`, `col-g1`, `col-g1-m4`; pack-only loop | ~2 h |
| desk, day 3 | read series 1, write §8.1 here | none |
| night 2 | series 2: `col-g2`, `col-g4`, plus `row` and the best of series 1 again (same build, same launches) | ~1.5 h |
| desk, day 4-5 | answer (§8), brief (PDF + LaTeX), links from #48 and #44 | none |

Board runs only with the user's go, and timing runs only on an **idle** board (2026-10-22): no other user's process at ≥ 20% of a core before each launch; if busy, wait (midnight) or use the other board; a run that saw a load is redone, not annotated.

## 7. Closure

- [ ] Spec #55 built and its tests green (host; the on-board pack test on `jbpi`).
- [ ] `row-intr` matches `row` within the launch spread on every tile and shape (the control holds), or the difference is written down and used.
- [ ] Series 1 and 2 run, 3 launches each, logs committed, RED NOTE or none stated.
- [ ] Pack-only loop run for every variant on B3 and B3_K256 at 16x8 and 16x15.
- [ ] Table per tile and shape: GOPs by mean time and by median rep, slow-rep share, per variant; uninstrumented, reference protocol, core 4, `--no-nc-overrides`.
- [ ] Written answer: which of A, B, C sets off the state (or none), whether the packer alone can, and whether any variant is a B3 win that keeps B2 and Reuse24.
- [ ] Brief (PDF + LaTeX) linked from this issue, #55, #48 and #44.

## 8. Results

### 8.1 Series 1 (2026-10-22, 19:40-23:48, both boards in parallel)

**Logs** (branch `issue55-pack-b-variants`, commit `383c886c`): `tests/pure_xdsl_blis_rvv/logs/issue55_2026-10-22_194043/`
(jbpi2, with `report_series1_jbpi2.md`) and `logs/issue55_jbpi/issue55_2026-10-22_194043/` (jbpi, with `report_series1_jbpi.md`).
**Build / protocol / core:** uninstrumented (`INSTRUMENTED=0`), reference protocol `--tmin 10`, core 4, `--no-nc-overrides`;
3 launches, left rotation; pinned tiles 16x6, 16x8, 16x15. **jbpi2 is the quoted board**; jbpi is the cross-check.
**Load:** jbpi2 clean (one 5-min wait at 22:04, overall idle 84.5%). jbpi: one GEMM run and one pack-only run were thrown out
(another user's `cc1plus` at 99%) and redone. RED NOTE, jbpi only: in run `run_2026-10-22_214815` (launch 2, `col-g1-m4`) core 3 was
at 72% during the 16x15 binary, and root `bluetoothd` sat at 40-50% of one core for most of the night. The jbpi launch-1 RED flags
in its report are a false alarm (`ps` lifetime average of an idle `sshd`; the `top` gate passed). None of the quoted numbers below
come from jbpi.

**B3 and B3_K256 on jbpi2.** GOPs by mean time / GOPs by median rep / slow-rep share:

| tile, shape | `row` (shipped) | `row-intr` | `scalar` | `col-gall` | `col-g1` | `col-g1-m4` |
|---|---|---|---|---|---|---|
| 16x15, B3 | 4.63 / 4.28 / 86% | 4.37 / 4.18 / 86% | 6.98 / 7.81 / 38% | 6.72 / 6.72 / 0 | 7.05 / 7.05 / 0 | 7.03 / 7.02 / 0 |
| 16x15, B3_K256 | 4.72 / 4.56 / 94% | 4.60 / 4.46 / 94% | 4.64 / 4.42 / 91% | 7.59 / 7.60 / 0 | 8.01 / 8.02 / 0 | 7.96 / 8.01 / 0 |
| 16x8, B3 | 3.83 / 3.65 / 84% | 3.50 / 3.30 / 93% | 6.89 / 7.45 / 9% | 7.11 / 7.12 / 0 | 7.05 / 7.09 / 0 | 7.07 / 7.11 / 0 |
| 16x8, B3_K256 | 3.65 / 3.55 / 96% | 3.76 / 3.63 / 93% | 7.39 / 7.91 / 14% | 6.69 / 6.69 / 0 | 7.88 / 7.90 / 0 | 7.87 / 7.90 / 0 |
| 16x6, B3 | 6.72 / 6.73 / 0 | 6.30 / 6.33 / 0 | 6.19 / 6.19 / 0 | **4.22 / 4.50 / 69%** | 6.90 / 6.95 / 0 | 6.89 / 6.94 / 0 |
| 16x6, B3_K256 | 4.81 / 5.33 / 51% | 4.33 / 4.42 / 63% | 5.87 / 6.47 / 41% | 4.38 / 4.59 / 48% | 7.61 / 7.58 / 0 | 7.72 / 7.66 / 0 |

jbpi gives the same slow/steady pattern on every cell. The steady `col-g1`/`col-g1-m4` cells agree to within 0.15 GOPs; cells that are
in the slow state move more, up to 0.43 GOPs (16x6 B3_K256 `row-intr`: 4.33 on jbpi2, 3.90 on jbpi)
(e.g. 16x15 B3: `row` 4.62, `col-g1` 6.90, `col-g1-m4` 6.96; 16x6 B3 `col-gall` 3.91 at 81% slow).

**Controls on jbpi2** (GOPs by mean time / by median rep; slow share 0 unless given):

| tile, shape | `row` | `row-intr` | `scalar` | `col-gall` | `col-g1` | `col-g1-m4` |
|---|---|---|---|---|---|---|
| 16x15, B2 | 12.64 / 12.68 | 12.33 / 12.35 | 12.13 / 12.17 | 12.38 / 12.40 | 12.43 / 12.45 | 12.57 / 12.60 |
| 16x8, B2 | 10.26 / 10.27 | 10.18 / 10.19 | 9.87 / 9.88 | 10.17 / 10.17 | 10.19 / 10.19 | 10.17 / 10.17 |
| 16x6, B2 | 10.24 / 10.25 | 10.05 / 10.06 | 9.87 / 9.87 | 10.19 / 10.19 | 10.13 / 10.14 | 10.15 / 10.16 |
| 16x15, R24 | 8.48 / 9.01 (17%) | 7.55 / 7.26 (53%) | 9.25 / 9.27 | 8.74 / 8.74 | 8.86 / 8.86 | 8.87 / 8.87 |
| 16x8, R24 | 8.95 / 8.97 | 7.07 / 7.08 (73%) | 9.03 / 9.11 | 8.99 / 8.99 | 9.17 / 9.16 | 9.15 / 9.15 |
| 16x6, R24 | 9.21 / 9.22 | 8.89 / 8.89 | 8.84 / 8.85 | 9.10 / 9.16 | 9.28 / 9.29 | 9.29 / 9.30 |

**Pack-only loop on jbpi2** (no compute between packs; slow-rep share, and mean pack time of fast / slow reps in ms):

| tile, shape | `row-intr` | `scalar` | `col-gall` | `col-g1` | `col-g1-m4` |
|---|---|---|---|---|---|
| 16x15, B3 | 91% (0.64 / 1.20) | 84% (0.66 / 1.60) | 0 (1.40) | 0 (0.89) | 0 (0.82) |
| 16x15, B3_K256 | 33% (0.38 / 0.44) | 95% (0.42 / 0.78) | 0 (0.66) | 0 (0.41) | 0 (0.36) |
| 16x8, B3 | 91% (0.88 / 1.17) | 5% (0.64 / 0.91) | 0 (1.08) | 6% (0.61 / 0.84) | 6% (0.56 / 0.79) |
| 16x8, B3_K256 | 0 (0.22) | 0 (0.38) | 0 (0.33) | 7% (0.29 / 0.46) | 5% (0.25 / 0.41) |

**What series 1 says:**

1. **Reading B down each column removes the slow state on B3.** `col-g1` and `col-g1-m4` have 0 slow reps on all six B3/B3_K256
   cells, on both boards. On the shipped tile, B3 goes from 4.63 to 7.03-7.05 GOPs by mean time (4.28 to 7.02-7.05 by median rep).
   That is above 16x6 with the shipped packer (6.72), the best steady B3 tile so far (#48).
2. **The packer alone can set off the state.** `row-intr` in the pack-only loop is 91% slow on B3 at 16x15 and 16x8, with no compute at all.
3. **The number of columns read side by side (property B) is not the trigger by itself.** `col-gall` reads all NR columns side by side and
   is steady at NR=8 and NR=15. But it is **69% slow at NR=6** (81% on jbpi), so column count matters in some way that does not go up
   with NR. Series 2 (`col-g2`, `col-g4`) is where this gets settled.
4. **Scalar loads are not safe either (property A alone is not it).** `scalar` is 84-95% slow in the pack-only loop at 16x15 and 91% slow in
   the full GEMM on 16x15 B3_K256. Its steadiness in #48 depends on the tile. What `row`, `row-intr` and `scalar` have in common is
   **4 bytes per column per touch (property C)**. Every variant that reads 32 B or more per column per touch is steady, except `col-gall` at NR=6.
5. **The control only partly holds.** `row-intr` matches `row` on the B3 cells (within 0.35 GOPs, same slow share). But it is lower on 16x6
   B3 (6.30 against 6.72, both steady), and it goes slow on Reuse24, where `row` does not: 73% at 16x8 and 53% at 16x15. So the C-intrinsics
   copy of the shipped read order is not the same as the xDSL one. The `col-g1` gains hold against both.
6. **Cost on the controls is small.**
   - B2: `col-g1-m4` is 0.07-0.09 GOPs under `row` (16x15: 12.57 against 12.64; 16x8: 10.17 against 10.26). `col-g1` is up to 0.21 under.
   - Reuse24: `col-g1-m4` is 8.87 at 16x15, against `row`'s 8.48 by mean time and 9.01 by median rep.
   - In a fast rep, a down-the-column pack costs more than `row-intr` (16x15 B3: 0.82 ms against 0.64 ms). But it never goes slow,
     while `row-intr` is slow 91% of the time at 1.20 ms.

**For series 2:** `col-g2`, `col-g4`, `row`, `col-g1-m4` (best of series 1), and `col-gall` again (to recheck its NR=6 result). Same build, jbpi2, idle gate.

### 8.2 Series 2 (2026-10-23, 10:01-13:11, jbpi2 only)

**Logs** (branch `issue55-pack-b-variants`, commit `bf6f2366`): `tests/pure_xdsl_blis_rvv/logs/issue55_2026-10-23_100122/` with
`report_series2_jbpi2.md`. **Build / protocol / core:** as §8.1 (uninstrumented, reference protocol `--tmin 10`, core 4,
`--no-nc-overrides`, 3 launches, left rotation). **Load:** none. 63 of 63 runs ok, no idle waits, no other user in any provenance,
and no other busy core in the per-core monitor.

**B3 and B3_K256 on jbpi2.** GOPs by mean time / GOPs by median rep / slow-rep share:

| tile, shape | `row` (shipped) | `col-g1-m4` (1 col) | `col-g2` (2 cols) | `col-g4` (4 cols) | `col-gall` (all NR) |
|---|---|---|---|---|---|
| 16x15, B3 | 4.53 / 4.28 / 89% | 7.01 / 7.00 / 0 | 6.48 / 6.48 / 0 | 7.17 / 7.17 / 0 | 6.67 / 6.67 / 0 |
| 16x15, B3_K256 | 4.94 / 4.67 / 89% | 7.88 / 7.88 / 0 | 7.86 / 7.91 / 0 | 7.57 / 7.59 / 0 | 7.57 / 7.58 / 0 |
| 16x8, B3 | 3.61 / 3.34 / 91% | 7.08 / 7.11 / 0 | 6.51 / 6.55 / 0 | 7.11 / 7.13 / 0 | 7.10 / 7.11 / 0 |
| 16x8, B3_K256 | 3.65 / 3.56 / 96% | 7.92 / 7.94 / 0 | **3.89 / 3.89 / 0** | 8.18 / 8.21 / 0 | 6.65 / 6.66 / 0 |
| 16x6, B3 | 6.70 / 6.71 / 0 | 6.88 / 6.93 / 0 | 6.36 / 6.39 / 0 | 6.48 / 6.53 / 0 | **3.88 / 3.77 / 84%** |
| 16x6, B3_K256 | 5.49 / 6.74 / 34% | 7.70 / 7.65 / 0 | **3.48 / 3.47 / 0** | **3.32 / 3.01 / 82%** | **4.24 / 4.47 / 54%** |

**Controls:** every variant is within 0.15 GOPs of `row` on B2 at all three tiles (16x15: 12.40-12.49 against 12.47), and none goes slow.
On Reuse24, `row` at 16x15 is 14% slow (8.68 by mean time, 9.14 by median rep). `col-g1-m4` is 8.88 / 8.89 with no slow reps.

**Pack-only on jbpi2** (median pack time in ms; slow-rep share 0 unless given):

| tile, shape | `col-g1-m4` | `col-g2` | `col-g4` | `col-gall` |
|---|---|---|---|---|
| 16x15, B3 | 0.84 | 1.14 | 1.04 | 1.37 |
| 16x15, B3_K256 | 0.36 | 0.48 | 0.50 | 0.67 |
| 16x8, B3 | 0.56 (6%) | 0.87 | 0.79 | 1.09 |
| 16x8, B3_K256 | 0.25 (4%) | **0.64** | 0.34 | 0.32 |

**What series 2 adds:**

1. **The repeats hold across nights.**
   - `col-g1-m4` matches series 1 to within 0.08 GOPs on every B3 cell.
   - `col-gall` at NR=6 is slow again: 84% on B3, 54% on B3_K256.
   - `row` keeps the same slow cells. Its slow share moves between nights on two cells: 16x6 B3_K256 is 51% then 34%, and 16x15 B3_K256 is 94% then 89%.
2. **Only one column at a time is safe everywhere.** `col-g1` and `col-g1-m4` are the only variants with no bad cell, across 2 series, 2 boards, 3 tiles and 2 K.
   Every grouped read (2, 4, all) has at least one bad cell, and the bad cells do not line up with group size:
   - `col-g2`: bad at 16x8 K256 and 16x6 K256
   - `col-g4`: bad at 16x6 K256 only
   - `col-gall`: bad at 16x6, both K
3. **There is a third state: steadily slow.** `col-g2` at 16x8 B3_K256 runs at 3.89 GOPs with a launch spread of 0.02 and no two-speed split.
   Its pack alone is uniformly slow too: 0.64 ms, against 0.34 for `col-g4` and 0.25 for `col-g1-m4`, with no slow reps.
   The same holds for 16x6 B3_K256 (3.48, 0 slow).
   The two-speed splitter cannot see this state, so **"0 slow reps" does not mean fast**. Read the GOPs column with it.
   The pack accounts for only about 0.4 ms of the roughly 1.6 ms lost per GEMM, so the compute is slowed too.
4. **With compute between packs, a grouped read can still set off the slow state, but alone it does not.** In the pack-only loop, every
   grouped variant has 0 slow reps. Its bad cells appear only in the full GEMM (or, for `col-g2` at K=256, as a uniformly slow pack).

### 8.3 Answer so far (series 1 + 2)

- **Property C sets off the slow state.** All three packers that read 4 B per column per touch go slow on B3, in the full GEMM and in the
  pack-only loop: `row`, `row-intr` and `scalar`. So the B packer on its own can trigger it. How a read is issued (property A) is not the
  cause: scalar loads go slow too.
- **Property B is not a simple cause.** Reading 2, 4 or all columns side by side, 32 B each, removes the 4-byte trigger but adds bad
  cells that depend on tile and K (NR=6 at both K, and K=256 for `col-g2`). The bad cells do not grow with the column count.
  They look like a placement effect, which fits #49's non-monotonic `ldb` result, rather than a stream-count limit.
- **A B3 fix that keeps B2 and Reuse24: read one column at a time (`col-g1-m4`).** On the shipped 16x15 tile, B3 goes from 4.53-4.63 to
  7.01-7.03 GOPs by mean time (4.28 to 7.00-7.02 by median rep). B3_K256 goes from 4.72-4.94 to 7.88-7.96. B2 stays within 0.1 GOPs and
  Reuse24 loses its slow reps. This is the C-intrinsics version; shipping it means a port to xDSL, which is out of scope here.
