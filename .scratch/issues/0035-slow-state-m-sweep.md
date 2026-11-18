# Issue #51: How small does M have to be? Sweep M at fixed tiles to separate rows from tile (sub-issue of #48)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/51

**Comes from:** #48 §8.5 (answer 5: "rows matter too": 32x7 is steady on the big shapes and 83-87% slow on B3) and #47 (the slow state
shows up in every shape with M up to 384 that ran on 16x15 and in none with M of 1000 or more, which all ran on 32x6/32x7).

**Role / Label:** `needs-triage` — sub-issue of #48, priority 3 of the #48 follow-ups (opened 2026-10-14). Needs a new dataset file and a pinned-tile config, no generator change.
**Status:** ✅ Closed 2026-11-18 (housekeeping: answered, results merged). Was: 🟢 Answered (§9, below). Board time: `jbpi` only, 25/25 runs, no red note (two earlier attempts on the same series were contaminated by another user's job and discarded — see §9).
**Target:** `jbpi` and `jbpi2`, core 4, quiet window, house rules of #39.
**Shape(s):** new: M in {64, 128, 192, 256, 384, 512, 1000} with N=384 and K=384, and the same M at K=256. B2 as the steady control.
**Primary Sources:**
- [[0031-slow-state-survey-all-shapes]] (#47) — the shapes that go slow and those that do not; tile and rows not separated.
- [[0032-b3-microkernel-dimension-sweep-pack-and-compute]] (#48) — B3 per tile (M=64 only).
- [[0029-b1b5-32x6-vs-32x7]] — the big shapes on 32x6 and 32x7.
- `tests/pure_xdsl_blis_rvv/datasets/issue46_ref_b3.dat` — the format of a small shape list.
- `get_optim_mc_nc_kc` in `generate_pure_blis_project.py` — mc and kc depend on M and the tile.

---

## 1. Question

#47 could not tell whether the slow state needs a small M or a particular tile, because every small-M shape ran on 16x15 and every big one on
32x6/32x7. #48 swept the tile at M=64 only. **At a fixed tile, at what M does the slow state stop, and is that M the same for every tile?**
The answer tells whether the shapes of the real models (M of 1000 and more) can fall into it on any tile, which decides how much B3's
problem matters outside B3.

## 2. Why it matters (why priority 3)

- If the slow state needs M up to a few hundred whatever the tile, B3 and the Reuse shapes are a corner. A tile picked for them (16x6, #48 §8.6) does not have to work for the big shapes.
- If some tile goes slow at large M too, the tile choice for the real models is at risk and needs the same sweep.

## 3. Plan

1. **Shapes.** A new dataset: N=384, K=384, M in {64, 128, 192, 256, 384, 512, 1000}; the same at K=256; B2 as the control.
2. **Tiles** (pinned rows, vector packer): 16x6 and 32x6 (steady on B3), 16x8 and 16x15 (two-speed on B3, 16x15 is the shipped one), 32x7 (two-speed on B3, steady on the big shapes).
3. **Record mc and kc of every run** (the reps logs do). `get_optim_mc_nc_kc` may give a different mc as M grows, and then mc moves with M.
   If it does, add a second row at a fixed `mc` (64) for the largest M, so M and mc come apart.
4. **Builds.** Uninstrumented, reference protocol, `--tmin 10`, `--no-nc-overrides --cpu 4`; 3 launches per board, A-B-B-A between the tiles.
5. **Read** with `reps_report.py`: per tile, a row of slow-rep share against M (and GOPs by mean time and by median rep), for both K.

## 4. How to read it

- **The slow share drops to 0 at the same M for every two-speed tile:** the state needs few A panels (or a small `mc` x `kc` A block). Note the M, and which of M, mc or the number of A panels it follows.
- **It drops at a different M per tile:** rows and tile work together. Put the threshold M per tile next to the layout result of [[0033-b-stream-layout-k256-flip]].
- **Some tile stays slow at M=1000:** the real-model shapes are at risk on that tile. Flag it for the tile choice (#48 §8.6) and for issue #2's matrix.

## 5. Out of scope

Changing the shipped tiles or the packer; the real-model shapes themselves (only after this says whether they are at risk); the reference design.

## 6. Dataset, configs and the mc/M desk check (2026-10-14)

`datasets/issue51_m_sweep.dat`: M in {64, 128, 192, 256, 384, 512, 1000} at N=384/K=384, the
same M set at K=256, and B2 as the control (15 shapes). One pinned-tile config per tile
(`config_issue51_16x6.json`, `_32x6`, `_16x8`, `_16x15`, `_32x7`), each a single row over that
one dataset — not bundled into one multi-row sweep, so tile order can really rotate between
launches (see §7).

`get_optim_mc_nc_kc` (`cache_config.py`), run at desk for every (tile, M, K) point:

| tile | mc at M = 64 .. 1000 (K=384, same at K=256) | kc |
|---|---|---|
| 16x6 | 64, 128, 192, 256, **384, 384, 384** | 256 |
| 16x8 | 64, 128, 192, 256, **384, 384, 384** | 256 |
| 32x6 | 64, 128, 192, 256, 384, 512, 768 | 128 |
| 16x15 | 64, 128, 192, 256, 384, 512, 768 | 128 |
| 32x7 | 64, 128, 192, 256, 384, 512, 768 | 128 |

`kc` never moves with M on any tile (good: it stays out of the way). `mc` does move with M —
it tracks `M` exactly until a per-tile ceiling, then holds flat. **For 16x6 and 16x8 the
planned M values already separate M from mc for free**: mc flattens at 384 from M=384 onward,
so M=384, 512 and 1000 share the same mc=384 — three points where M keeps growing but mc does
not, with no extra row needed. **For 32x6, 16x15 and 32x7, mc keeps climbing across the whole
planned range** (768 at M=1000, not yet flat), so M and mc stay coupled for those three tiles
within this sweep.

Plan step 3 asks for "a second row at a fixed mc (64) for the largest M" if mc moves with M.
Building that for the three uncapped tiles would need a new mc override in the driver (there is
today only `RVV_FORCE_NC` for `nc`, nothing for `mc`) — a generator/driver change, which this
issue's own Role/Label line says is out of scope ("no generator change"). Left as a known gap
rather than silently built: if 32x6/16x15/32x7 turn out to need M and mc told apart, that is a
small follow-up spec, not a change to fold in here.

## 7. The series script (2026-10-14)

`run_issue51.sh` runs the plan of §3: one launch runs all five tiles in rotating order (with
tiles a b c d e, launch 1 is a b c d e, launch 2 is b c d e a, launch 3 is c d e a b, and so
on), each tile its own config/project, both boards taking turns, the shared lock and RED NOTE
conventions of `run_issue46.sh`/`run_issue48.sh`/`run_issue49.sh`. Each tile is a separate
single-row config specifically so the rotation is real, rather than bundled into one multi-row
sweep whose internal row order cannot rotate — the gap #50's code review found and #50 itself
could not avoid without rebuilding its sweep machinery.

**Code review note: "A-B-B-A order between the tiles" (§3.4) is not literally what this does.**
That phrase is house rule 2's two-symbol alternation pattern; with five tiles there is no clean
two-symbol reading, so this cyclically rotates all five instead, the same choice made for
[[0033-b-stream-layout-k256-flip]]'s four pads. **A first pass default of 3 launches (house
rule 2's bare minimum) was wrong here and was corrected after review**: with 5 tiles and only 3
launches, a cyclic rotation never puts two of the five tiles — including 16x15 and 32x7, the two
this issue most wants isolated (§6) — in the first slot, and a different two never in the last
slot, which defeats the actual point of rotating at all. The default is now **5 launches**
(matching the default tile count), which closes a full cycle: every tile sits first once and
last once. `test_run_issue51.py` checks this directly
(`test_the_default_five_launches_put_every_tile_first_and_last_once`). A custom `--tiles` with a
different count should get a matching `--launches`.

Validated with `--dry-run` and against a stubbed board (`test_run_issue51.py`, 13 tests green)
— not yet run against the real boards: this host's jump host (`fej`) does not resolve from
this session. Run it from a host that can reach the boards:

```bash
setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue51.sh --start-at 00:00 \
    > tests/pure_xdsl_blis_rvv/logs/issue51.out 2>&1 < /dev/null &
```

## 9. The board result (2026-10-14)

`run_issue51.sh --boards jbpi` ran the plan: 5 launches (full tile-rotation cycle), 25/25 runs,
clean (`issue51_2026-10-14_215835`). **One board only, not two** — `jbpi2` had another user's
job pinned at ~100% throughout; the cross-board check is still open. Two earlier attempts at
this same series (`issue51_2026-10-14_150156`, `issue51_2026-10-14_181219`) picked up that same
job wandering onto `jbpi` partway through (launches 3-5 both times) and are discarded — not
cited below; they are kept in `logs/` with their own RED NOTEs for the record.

Slow-rep share by M, `reps_report.py` per tile (5 launches each), B2 control 0% everywhere:

| tile | K | M=64 | 128 | 192 | 256 | 384 | 512 | 1000 | mc at these M |
|---|---|---|---|---|---|---|---|---|---|
| 16x6 | 384 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 64,128,192,256,**384,384,384** |
| 16x6 | 256 | 77 | 62 | 60 | 51 | 49 | 35 | **0** | (same) |
| 32x6 | 384 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 64,128,192,256,384,512,768 |
| 32x6 | 256 | 34 | 58 | 37 | **0** | 0 | 0 | 0 | (same) |
| 16x8 | 384 | 93 | **0** | 0 | 0 | 0 | 0 | 0 | 64,128,192,256,**384,384,384** |
| 16x8 | 256 | 0 | 0 | 0 | 0 | 0 | 0 | **33** | (same) |
| 16x15 (shipped) | 384 | 84 | 58 | 65 | 45 | 23 | **0** | 0 | 64,128,192,256,384,512,768 |
| 16x15 (shipped) | 256 | 92 | 95 | 92 | 85 | 81 | 17 | **0** | (same) |
| 32x7 | 384 | 63 | **0** | 0 | 0 | 0 | 0 | 0 | 64,128,192,256,384,512,768 |
| 32x7 | 256 | 76 | 75 | 61 | 53 | 28 | **0** | 0 | (same) |

**Answer: the threshold M is not one value — it moves with both tile and K, sometimes by a lot
(16x6/K256 needs M=1000 to clear; 32x6/K256 clears by M=256) — so rows and tile work together,
as the plan's second reading bullet expected. But mc alone does not explain it either: 16x6 and
16x8 cap `mc` at 384 from M=384 onward (§6), yet 16x6/K256's slow share keeps falling across
M=384→512→1000 (49%→35%→0%) at the *same* `mc` — raw M (or the number of ic-iterations, 1→2→3
at this mc) still moves the needle after `mc` itself stops growing.**

**The important result is 16x8/K256: it is clean (0% slow) from M=256 through M=512, then goes
33% slow at M=1000 — the only case in this sweep where the state *reappears* at large M after
being clear.** Same `mc` (384, saturated) at M=384, 512 and 1000; only M=1000 is slow. This is
exactly the plan's third reading bullet ("some tile stays slow at M=1000: the real-model shapes
are at risk on that tile"): **16x8 at K=256 is not safe for a large-M real-model shape**, and
should be flagged against the tile choice (#48 §8.6) and any real-model matrix (issue #2),
alongside the already-known risk that 16x6/16x15/32x7 at K=256 need M up to 512-1000 to clear
at all. The only combinations clean at every M tried, both K: 16x6/384, 32x6/384 (and 32x6/256
from M≥256).

## 8. Closure

- [x] Dataset and pinned-tile config committed, mc and kc of every point printed. See §6.
- [x] Runs — `jbpi` only (not `jbpi2`, occupied throughout), 5 launches, logs committed, no red note (after two discarded contaminated attempts). See §9.
- [x] Table of slow-rep share against M per tile, for both K, written here. See §9.
- [x] Written answer: the threshold M differs by tile and K and does not reduce to `mc` alone; 16x8/K256 re-enters the slow state at M=1000 after being clean at smaller M — flagged as a real-model risk. See §9.
- [ ] `jbpi2` cross-check, once free.
