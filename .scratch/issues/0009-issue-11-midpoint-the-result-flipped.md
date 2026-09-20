# Issue #11 midpoint: the result flipped sign, and what that does to the remaining sub-issues

**Role / Label:** `assessment` — no code changed; this reads the run of 2026-09-19 18:54 against #11's own predictions
**Status:** 🟡 Addendum 2026-09-20 (§8): 16 pinned runs on two boards. 7 of 14 sub-issues closed. This note argues the remaining 7 should mostly **not** be implemented, and says what to do instead.
**Target:** `jbpi` (BananaPi M2 Pro, VLEN=256) via `fej`. Uninstrumented build (`INSTRUMENTED=0`), reference protocol, `--no-nc-overrides`.
**Primary Sources:**
- GitHub #11 and its children #12–#25; #26 (pack swap) remains separate.
- `logs/run_2026-09-19_185453/` — the run this note is about.
- `logs/refbase_2026-09-18_164740/` — the only reference data that exists.
- [[0007-timer-tax-measured-not-estimated]], [[0008-ref-protocol-warmup-effect-measured]].
- [[project_benchmark_reference_comparison_trap]] — the numbers below are KerGOPs-free; all are pack-inclusive.

---

## 1. The headline moved from "a few percent behind" to "13–14% ahead"

#11 closed with a prediction, under *Where the numbers actually land*:

> Correcting B1 for pack inclusion **and** the per-tile timer gap puts us at roughly
> **6.7–6.9 GOPs against the reference's 7.02** — parity to a few percent behind.

Measured, same board, same shapes, pack-inclusive on both sides:

| case | #11 predicted | refbase 09-18 | **run 09-19 18:54** | vs reference |
|---|---|---|---|---|
| B1 `1024x384x1024` 32x6 | 6.7–6.9 | 7.02 | **8.04** | **+14.5%** |
| B5 `1024x384x4096` 32x6 | — (6.68 then) | 7.06 | **8.00** | **+13.3%** |

Per-call time on B1 went 120.481 ms → 100.109 ms, a 16.9% cut. On B5, 493.731 ms → 402.528 ms,
18.5%.

**This inverts the conclusion #11 was written to support.** Every remaining sub-issue has to be
re-ranked against a favourable claim rather than an unfavourable one, because a +14% result is
attacked harder than a −2% one.

## 2. The timer tax was ~6x larger than the audit's model

#11 finding 3 built a ns/tile model from `RVV_Time_ms − (Vector_Pack_Time_ms + Compute_Time_ms)`
and predicted the uninstrumented build would come in **2–3% faster on B1**. Step 4 wrote down the
falsifier:

> If it doesn't, the ns/tile model above is wrong and worth revisiting.

It didn't. B1 gained ~17%, not 2–3%.

**Why the model missed it.** The gap it measured is only the part of the tax landing *between*
tiles. Anything charged *inside* `Vector_Pack_Time_ms` or `Compute_Time_ms` cancels in that
subtraction and is invisible to it. The `perf_event_open` reads added in #8 sit inside those
windows. So the subtraction could only ever see the chrono pairs' outer half, and the model was
built on the visible fraction.

Consequence for anything else derived the same way: the fringe-tile cost table (finding 4) and the
pack-percentage figures all come from instrumented runs and inherit this distortion. They are
upper bounds on *instrumented* behaviour, not measurements of the shipped build.

## 3. Sub-issues #18 and #19 bought nothing measurable

Six runs today, B1 mean-time GOPs, with the commits interleaved:

| run | after | B1 | B5 | S1 |
|---|---|---|---|---|
| 12:25 | (uninstr. baseline) | 7.92 | 7.81 | 8.57 |
| 17:19 | — | 7.91 | 7.82 | 8.47 |
| 17:55 | #27 protocol default | 8.01 | 7.99 | 8.48 |
| 18:15 | #16 no-nc-overrides | 7.94 | 7.89 | 8.44 |
| 18:41 | **#18 pad memsets removed** | 7.94 | 7.85 | 8.59 |
| 18:54 | **#19 Ac/Bc at KR=1** | 8.04 | 8.00 | 8.44 |

B1 sits in a 7.91–8.04 band (1.6% wide) across the whole day with no trend. #18 moved it by
exactly zero. #19's +1.3% is inside the band and inside the day's noise.

Both were real differences from the reference, correctly identified, correctly fixed — and worth
nothing on the clock. That is the base rate to price the *remaining* structural fixes at.

## 4. What #11 did not find: the reference has no odd-NR kernel

`ext/RVV_ukernels_benchmark/kernels/RVV_256_LDX/fp32/` ships 96 tile shapes:
MR ∈ {8,16,24,32,40,48} × NR ∈ {2,4,…,32}. **NR is even, always.**

Our three pinnings are 16x15, 32x6, 32x7. Only **32x6** exists on the reference side. That is why
`refbase` covers B1 and B5 and nothing else, and why the `Ref GOPs` column in
`run_2026-09-19_185453/comparison.md` is `N/A` on 8 of 10 rows.

#11 knew B1 was "exact ... no bracketing" and picked it as the probe shape, but never drew the
consequence: **the head-to-head can only ever be exact on 32x6, which is 2 of our 10 cases.** Our
headline square shapes S1–S5 run at 32x7, where the reference has nothing to compare against.

Two readings, both worth stating:

- *Against us:* the parity claim rests on 2 shapes. S1–S5 have no reference number at all.
- *For us:* being able to emit NR=7 and NR=15 is one of the two intended differences doing its
  job. The reference's table is fixed and even-only; the generator is not. That is a result, not
  an inconvenience — but it has to be framed as best-tile vs best-tile, never as same-tile.

## 5. Where the seven open sub-issues now rank

**Tier 1 — the +14% is not defensible without these.**

- **#21 (pinning, back-to-back)** — now the largest hole. `taskset` is still absent from both
  `compile_macro_bench.sh` and `ref_baseline/run_ref_baseline.sh`. Our number is 09-19 18:54;
  the reference's is 09-18 16:47. Unpinned, shared board, 26 hours apart, and B3 shows a 35% CV
  inside a single run, so the board is demonstrably noisy. A 14% claim resting on a cross-day
  unpinned pair will not survive a reviewer.
- **#24 + #17 (correctness)** — every row still logs `SKIP`. MR=32 is the pinning that produces
  B1, B5, B4 and S1–S5, and its 3rd and 4th vector pack chunks have never been checked against
  the oracle. We are claiming a speed win on output that has not been verified on hardware. This
  is not a performance item; it is whether the result is real.
- **`OPT_MODE` default** (in #11's body, not a sub-issue) — `run_ref_baseline.sh:26` still reads
  `OPT_MODE="OPT"`. With the claim now favourable, the right move is not the audit's recommended
  flip to LDX but reporting **both**: LDX for like-for-like, OPT because it is the reference at
  its strongest. Beating the strongest form is the claim worth making.
- **Reference coverage** (unnumbered, see §4) — decide between bracketing 32x7 against 32x6/32x8
  with the caveat written down, or generating a 32x6 variant of S1–S5 so every case has an exact
  counterpart, reporting our best tile separately.

**Tier 2 — cheap, do them in the same session as Tier 1.**

- **#20 (estimator)** — one-line fix, and with the delta now large it is the kind of detail that
  gets picked at. Best-of-K on both sides.
- **#23 (n=1 marker)** — documentation only, and it guards a number that runs *against* the
  reference, so it costs us nothing to disclose and buys credibility.

**Tier 3 — do not implement; document and close.**

- **#22 (-O2 → -O3)** — already measured at 146 instructions either way on the hot kernel, ~2% on
  the packers. #18 and #19 were argued to be worth more than this and delivered zero. Align the
  flag if it is free, but do not spend a board run proving it.
- **#25 (cleanups)** — correctness-of-code items, not measurement items. The layout assert and
  the dead-code removal are worth doing whenever the file is next open. None of it moves a number.

**Separate thread, and the only real perf question left:**

- **#26 (B3 pack swap)** — B3 is the one genuinely broken case: 4.77 / 4.91 / 5.26 / 5.71 / 5.83 /
  6.34 across today's six runs, 35% CV within a single run, per-rep range 3.97–9.72. That is not a
  comparison artifact and not a parity question. It is the last open engineering problem.

## 6. Recommendation

Stop implementing parity fixes. The two most recent ones moved nothing, and the remaining ones are
smaller than those were. The gap between us and the reference is no longer made of unfixed
structural differences — it is made of **an unpinned cross-day measurement on 2 shapes with no
correctness check**.

Sequence:

1. `taskset` on both sides, provenance captured (#21).
2. Correctness on: fix the `gemm_ref` braces, `SKIP_REF_RUN` off on B2-sized shapes, pack test
   parameterised over `[6, 7, 15, 16, 32]` (#17, #24).
3. One session: reference at OPT **and** LDX, our side pinned, back to back, and a decision on
   32x7 coverage per §4.
4. Then write up #11 with the result it actually produced, not the one it predicted.

Everything in Tier 3 becomes a paragraph in that write-up.

## 7. Open questions

- Is the ~17% attributable to instrumentation alone? Not isolated — #14, #16, #18, #19 and #27 all
  landed between refbase and now. §3 rules out #18 and #19; #16 does not touch B1 (no override on
  that shape). That leaves #14 and #27, and §2 argues #14 carries it. An instrumented-vs-not pair
  on B1 in one session would settle it and is cheap.
- Does the reference's own number move under pinning? If it gains more than we do, the +14%
  shrinks. Unknown until step 1 is done, and the honest position until then.

## 8. Addendum 2026-09-20: 16 runs, two boards, pinned, against both references

**What ran.** 16 runs, 8 on `jbpi` and 8 on `jbpi2`, taking turns every 30 min from 23:00 on 19 Sep to 06:30 on 20 Sep. Uninstrumented, reference protocol `tmin=10 s`, `--no-nc-overrides --cpu 0`. Each took about 7 min. Logs: `logs/run_2026-09-19_230648/` through `logs/run_2026-09-20_063652/`. The schedule and why the runs no longer collide: [[0012-cron-two-boards-share-one-directory]]. Interactive view of all of it: https://claude.ai/artifact/ANoEN8zBCAehAzrQN5rXbE (private; its source is not in the repo).

All numbers are 2·m·n·k ÷ mean wall time of the whole GEMM call, packing included (`RVV_Time_ms`). Both boards' means are of 8 runs each.

| layer | config | reps per run | jbpi | jbpi2 | both | past publication | vs it | fresh reference run | vs it |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 32x7 | 43 | 8.51 | 8.47 | 8.49 | 8.5 | −0.1% | | |
| S2 | 32x7 | 6 | 8.77 | 8.72 | 8.75 | 8.7 | +0.5% | | |
| S3 | 32x7 | 2 | 9.07 | 9.03 | 9.05 | 8.9 | +1.7% | | |
| S4 | 32x7 | 1 | 8.84 | 8.83 | 8.83 | 9.0 | −1.8% | | |
| S5 | 32x7 | 1 | 9.18 | 9.15 | 9.16 | 9.0 | +1.8% | | |
| B1 | 32x6 | 100 | 8.09 | 8.00 | 8.04 | 8.0 | +0.6% | 7.02 | **+14.5%** |
| B2 | 16x15 | 6740 | 12.72 | 12.72 | 12.72 | 12.5 | +1.8% | | |
| B3 | 16x15 | 2808 | 5.34 | 5.26 | 5.30 | 8.8 | **−39.8%** | | |
| B4 | 32x7 | 25 | 7.83 | 7.78 | 7.80 | 8.5 | −8.2% | | |
| B5 | 32x6 | 25 | 7.86 | 7.78 | 7.82 | 7.5 | +4.3% | 7.06 | **+10.8%** |

"Past publication" is the user's table (see [[project_reference_baseline_perf_user_stated]]); "fresh reference run" is `logs/refbase_2026-09-18_164740/`, the only measurement of the reference design itself (B1 and B5 only).

**What it shows.**
1. **The two references disagree with each other by 14% on B1** (8.0 published, 7.02 measured). So "we are 14% ahead" and "we are at parity" are both true, depending on which reference is meant. §1's +14.5% on B1 reproduces (8.04 again). On B5 it is +10.8%, lower than §1's +13.3% (8.00 then, 7.82 now).
2. **Against the published table, 7 of 10 layers are within ±3%.** The three outside are B3 (−39.8%), B4 (−8.2%) and B5 (+4.3%).
3. **The two boards agree.** The largest gap between their means is 1.2% (B1). No board effect worth chasing.
4. **B3 is far below both the published table and the archived sweep (8.53, −37.9%), and is the noisiest layer.** Its reps vary by 34% (CV) inside a run. In the one run I looked at, the median rep was 4.4 against a mean-time figure of 5.3, with single reps up to 9.9. The cause was not investigated here (issue #1). B2 is next at CV 3.6%; every other layer is at or under 1%.
5. **Pinned B1 varies 1.6% between runs on jbpi (8.02–8.15) and 2.4% on jbpi2 (7.86–8.05).** #21 asked for this measurement. There is no unpinned run at the same protocol to compare it against, so "tighter than before" is still unmeasured.

## 9. What §8 does not establish

- **No back-to-back reference.** The fresh reference run is from 18 Sep, unpinned, `tmin 3 s`, on jbpi2. Ours are pinned to core 0 and `tmin 10 s`. This is the "does the reference's number move under pinning?" question in §7 and it is still open, so the +10.8% and +14.5% stand on a cross-day pair.
- **The published table's origin is unknown to me** (board, build, timing region, pinning). Parity with it is only as strong as that table.
- **S3, S4 and S5 have 1 or 2 reps per run** at `tmin 10 s`, so they have no inside-run spread. Only their between-run range is meaningful.
- **B3's gap is unexplained.** It is the one layer where our result looks clearly worse than every reference.
- **Not in this addendum:** the five jbpi2 runs of 18 Sep (older warm-up protocol) and the OpenBLAS column of `ext/*_comparison.txt` (where those were measured is unchecked).

