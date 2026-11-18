# Issue #50: Prefetcher counters at the four tiles that flip: 16x6 and 16x7 at K=384, 16x6 and 16x8 at K=256 (sub-issue of #48)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/50

**Report:** [issue-48-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-48-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-48-brief.tex)), "Next steps" — updated 14 October 2026 with a pointer to this issue.

**Comes from:** #48 §8.6 ("count the prefetcher events at 16x6 against 16x7"), widened to the two B3_K256 tiles that flip the other way (#48 §8.3).

**Role / Label:** `needs-triage` — sub-issue of #48, priority 2 of the #48 follow-ups (opened 2026-10-14). No code change: `--instr --raw-events x60-l2` already exists (#42).
**Status:** ✅ Closed 2026-11-18 (housekeeping: answered, results merged). Was: 🟢 Answered (§8, below). Board time: `jbpi` only, 3/3 runs, no red note.
**Target:** `jbpi` and `jbpi2`, core 4, house rules of #39.
**Shape(s):** B3 (64x384x384), B3_K256 (64x384x256); B2 as the steady control.
**Primary Sources:**
- [[0032-b3-microkernel-dimension-sweep-pack-and-compute]] (#48) §8.2-8.4 — which tiles are steady and which are two-speed, per K.
- [[0025-b3-slow-state-root-cause]] (#41) §8.2-8.4 — the counter pattern of a slow rep: the same L2 accesses, 2-4x the demand L2 misses, about half the L1D prefetch hits and refills.
- [[0026-l2-counters-per-rep-spec]] (#42) — the per-rep raw counters (`--raw-events x60-l2`) and how they are read.
- [[0033-b-stream-layout-k256-flip]] (sub-issue of #48, priority 1) — the layout question this issue gives counter evidence for.

---

## 1. Question

The four smallest pairs that flip the slow state in #48 (vector packer, `jbpi` / `jbpi2` slow-rep share):

| shape | steady tile | two-speed tile |
|---|---|---|
| B3 (K=384) | 16x6: 0% / 0% | 16x7: 79% / 68% |
| B3_K256 | 16x8: 0% / 0% | 16x6: 48% / 58% |

**In each pair, do the prefetcher counters of the steady tile look like the fast reps of the two-speed tile, and do the two-speed
tile's slow reps show the #41 pattern (about half the L1D prefetch hits, 2-4x the demand L2 misses)?** And within one tile across
the two shapes (16x6 at K=384 against K=256), what moves in the counters when only K moves?

## 2. Why it matters (why priority 2)

- It ties the flips of #48 to the one mechanism #41 found. If the counters follow the state and not NR, the prefetcher backs off for a
  reason that depends on the layout ([[0033-b-stream-layout-k256-flip]]), not on the stream count.
- It costs no new code, only board time.

## 3. Plan

1. **Builds.** Instrumented (`--instr`, warm-up protocol, 1000 reps) with `--raw-events x60-l2`, vector packer, `--no-nc-overrides --cpu 4`.
   Tiles 16x6, 16x7, 16x8 and the shipped 16x15 as an anchor to #41's numbers; shapes B3, B3_K256 and B2 on each tile.
   (A pinned-tile config like `config_issue48_sweep.json`, cut to these four rows.)
2. **Repeats.** 3 launches per board, A-B-B-A order between the tiles, other users read before and after every run.
3. **Read** per tile and shape, as #41 §8.2 did: slow-rep share, then for fast reps and for slow reps, split into pack and compute:
   cycles, L2 loads, L2 load misses, L1D prefetch hits and refills, L1D misses, AR-channel stall cycles. Put the slow/fast ratio next to them,
   and add a column comparing the steady tile with the two-speed tile's fast reps.
4. **Check the instrumented build sees the same flips.** #48 §8.4: the instrumented build changes when the state starts (32x7 had no slow
   reps on `jbpi2` instrumented, 83-87% uninstrumented). If a pair does not flip in the instrumented build, write that down and read
   only the fast/slow split of the tile that does go slow; do not read the steady side as "steady" in that case.

## 4. How to read it

- **Steady tile counters = fast-rep counters of its pair, slow reps show the #41 pattern, at both K:** one mechanism for every flip. What sets it off is the layout (to be pinned down in [[0033-b-stream-layout-k256-flip]]).
- **The steady tile already has fewer prefetch hits per byte than the fast reps of its pair:** the prefetcher works differently per tile even when fast. Look at the per-tile pack cost of #48 §8.4 (16x6 packs slower in fast reps than 16x8) with that in mind.
- **A pair does not flip in the instrumented build:** the counters cannot see this pair. Fall back to the cycle count per rep of the uninstrumented logs.

## 5. Out of scope

Any change to the generator, the driver or the packers; the reference design; tiles beyond these four.

## 8. The board result (2026-10-14)

`run_issue50.sh --boards jbpi` ran the plan: 3 launches (`issue50_2026-10-14_143115`, 3/3 runs
ok, no red note). **One board only, not two** — `jbpi2` had another user's job pinned at
~100% throughout; the cross-board check is still open.

**Step 1 first: neither named pair actually flips in this instrumented (warm-up protocol,
1000-rep) build.** Slow-rep share per tile/shape (`reps_report.py` on each tile's own
`reps_*.txt.gz`, 3 launches):

| tile | B3 (K=384) | B3_K256 | B2 |
|---|---|---|---|
| 16x6 | 0% | 1% (1 of ~300 reps) | 0% |
| 16x7 | 0% | 15% | 0% |
| 16x8 | **65% slow** (4.89→4.69 GOps) | 0% | 0% |
| 16x15 (anchor) | **50% slow** | **54% slow** | 0% |

The two pairs this issue asked about (16x6 steady / 16x7 two-speed at K=384; 16x8 steady / 16x6
two-speed at K=256) both come back essentially one-speed here — exactly the #48 §8.4 caveat
this issue's own plan (step 4) anticipated ("the instrumented build changes when the state
starts"). Per that plan's own fallback: **for those two pairs, read the uninstrumented
cycle-count evidence already in #48/#49 instead** (#48: 16x6/16x7 at K=384 = 0%/79-68% slow;
#49 §11: 16x6/16x8 at K=256 = 61%/0% slow at pad0) — the counters here cannot see them.

**What this build does show a flip on — 16x8/B3 and the 16x15 anchor on both shapes — repeats
#41's exact pattern, tightly:**

| tile / shape | Pack_Cycles slow/fast | Compute_Cycles slow/fast | L1D prefetch hit, pack | L1D prefetch hit, compute | L1D miss, compute | L2 load miss, compute |
|---|---|---|---|---|---|---|
| 16x8 / B3 | 2.00x | 1.65x | 0.45x | 0.51x | 4.42x | 6.58x |
| 16x15 / B3 (anchor) | 2.04x | 1.52x | 0.41x | 0.51x | 3.01x | n/a (l2_load_miss not in this table; ar-stall 0.47x) |
| 16x15 / B3_K256 | 1.91x | 1.49x | 0.52x | 0.51x | 3.17x | n/a |

Compute-phase L1D prefetch hit ratio is 0.51x in all three — the same mechanism, not three
different ones. Pack takes about twice as long and gets roughly half its prefetch hits; demand
misses (L1D, L2) run 1.6–6.6x higher in the slow reps, in both pack and compute. This extends
#41's finding (previously one tile, 16x15/B3) to a second tile (16x8) and a second shape at the
anchor tile (16x15/B3_K256): wherever this build actually shows a flip, it is the same
prefetcher-backing-off mechanism, not a different one per tile or K.

**Answer:** the counters follow the state, not NR, everywhere this build can see a flip at all
— but it can't see the two pairs the issue specifically asked about, so the question "does the
steady tile of *those* pairs look like the two-speed tile's fast reps" stays open; the
uninstrumented reference-protocol numbers (#48 §8.2, #49 §11) are the only evidence for those
two, and they show the same ~2x slowdown by wall time without a counter breakdown.

## 6. Closure

- [x] Instrumented runs with raw events — `jbpi` only (not `jbpi2`, occupied throughout), 3 launches, logs committed, no red note. See §8.
- [x] Counter table per tile and shape, fast and slow reps, pack and compute apart, written here. See §8.
- [x] Written answer: counters follow the state wherever a flip shows up in this build (16x8/B3, 16x15 both shapes); the two named pairs don't flip here at all — fall back to #48/#49's uninstrumented numbers for them. See §8.
- [ ] `jbpi2` cross-check, once free.

## 7. The series script (2026-10-14)

`run_issue50.sh` runs the plan of §3: `config_issue50_sweep.json` (16x6, 16x7, 16x8, the shipped
16x15, cut from `config_issue48_sweep.json` to just these four rows), `--instr --raw-events
x60-l2` by default, 3 launches per board, both boards taking turns, the shared lock and RED NOTE
conventions of `run_issue46.sh`/`run_issue48.sh`. No code change (as §3.1 says: `--instr
--raw-events x60-l2` already exists, #42) — only the cut-down config and the wrapper are new.

Validated with `--dry-run` and against a stubbed board (`test_run_issue50.py`, 22 tests green
together with `test_run_issue49.py`) — not yet run against the real boards: this host's jump
host (`fej`) does not resolve from this session. Run it from a host that can reach the boards:

```bash
setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue50.sh --start-at 00:00 \
    > tests/pure_xdsl_blis_rvv/logs/issue50.out 2>&1 < /dev/null &
```

**Code review note: "A-B-B-A order between the tiles" (§3.2) is not implemented.** One
`compile_macro_bench.sh` call processes every row of `config_issue50_sweep.json` in the file's
own order (16x6, 16x7, 16x8, 16x15) on every launch — the same way `run_issue48.sh`'s own
`vec-instr` arm processes its 25-row config in file order every launch, with no per-tile
rotation either. Rotating tile order between launches would mean compiling and running each
tile as its own invocation (four separate builds per launch instead of one combined sweep),
which is a real cost increase over what §3.1 asked for ("a pinned-tile config... cut to these
four rows", i.e. one combined config). Left as a known gap rather than silently resolved: if
a launch-position bias (e.g. thermal drift as the board warms up) turns out to matter, redo
this as four rotating single-tile launches instead.
