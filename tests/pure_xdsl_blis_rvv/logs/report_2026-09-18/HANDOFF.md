# Nightly RVV benchmark report — handoff

**Live artifact (source of truth, already published, survives context clear):**
https://claude.ai/artifact/JtGVuhJrKp7SArnHn4V5Xc

Local copies in this directory (backup / for git, not required for the artifact to keep working):
- `report.html` — the published page's source, version 2 (bar-chart redesign)
- `build_report_data.py` — regenerates `bench_data.json` from the run logs
- `bench_data.json` — the aggregated dataset embedded in `report.html`

## What the report covers

5 hourly cron repeats on **jbpi2 only** (2026-09-18, 00:01–04:41). `jbpi` failed all
5 of its own scheduled runs last night with `Fatal error: can't create
gemm_b1_xdsl.o: Stale file handle` on-board — a remote filesystem fault on that
board, unrelated to scheduling. So there is currently no cross-machine
comparison, only run-to-run repeatability on one board.

Sections: average GEMM performance by layer (bar + jitter dots + min/max
whiskers), vector vs. scalar packing throughput, comparison against the
archived reference design (`ext/bert_large_comparison.txt` +
`ext/square_comparison.txt`, confirmed against a chart the user posted), and an
independent hardware-perf-counter section (IPC, stall %, full counter table).

## Pending requests (not yet applied — mid-edit when context was interrupted)

The user asked for three more changes to the same artifact, in this order of
what was said:
1. **Remove the OpenBLAS series** from all performance-comparison charts —
   applies mainly to the reference-design chart (`chartRef` in report.html),
   which currently has 3 series: `current` (this run), `archived` (xDSL ref),
   `openblas`. Drop the `openblas` series entirely (bar + legend chip).
2. **Richer hover tooltips**: hovering a bar should show the data point, the
   comparison point, the ratio, and the percentage — not just the single
   value currently in each `<title>` (native SVG tooltip). This needs a real
   interactive tooltip (mousemove-tracked div, not native `<title>`) that,
   for a hovered bar, also looks up the paired value from the other visible
   series in that category and computes ratio/delta%.
3. **Fix overlapping text near the top of a chart** — user reported text
   overlapping text, hard to read, "on top of the graph". Not yet diagnosed:
   this session has no headless browser available (checked for chromium,
   google-chrome, playwright — none installed) to screenshot and confirm
   exactly which text collides. Best next step is to actually open the live
   artifact URL above and look, or install a headless browser to check the
   `.cat-label` / `.cat-sub` / `.val-label` text elements in `report.html`'s
   `BarChart()` function for tight spacing (likely candidates: category label
   vs. shape sub-label at the bottom, only 12px apart; or the topmost
   y-axis gridline label sitting close to the chart's top edge).

## How to continue

Edit `report.html` directly (or regenerate `bench_data.json` first via
`python3 build_report_data.py` if the underlying run logs changed), then
republish with the Artifact tool using `url:
https://claude.ai/artifact/JtGVuhJrKp7SArnHn4V5Xc` to update in place — do not
publish without `url`, that creates a separate artifact.

Note: `/handoff` is not a recognized skill/command in this session (not in
the available-skills list), so it wasn't literally invoked — this file plus
the durable artifact link above is the handoff in its place.
