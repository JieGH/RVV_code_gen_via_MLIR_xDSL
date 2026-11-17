"""Issue #79 part B: recompute #63's numbers from the raw logs, without #63's scripts.

Reads all_raw_*.txt of our launches (RVV_Time_ms = mean time per call, KerGOPs_Median = median rep)
and his gemm.x CSVs (seconds per call). Prints Markdown.
Usage: recompute_63.py --ours8 RUN... --ours1 RUN... --his CSV...
"""
import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_ours(run: Path) -> dict[tuple[str, str], dict]:
    """(shape, tile) -> row of one launch"""
    f = next(run.glob("all_raw_*.txt"))
    out = {}
    for r in csv.DictReader(f.open()):
        tile = r["Variant"].replace("MR", "").replace("_NR", "x")
        out[(r["Layer"], tile)] = {
            "m": int(r["M"]), "n": int(r["N"]), "k": int(r["K"]),
            "ms": float(r["RVV_Time_ms"]), "med": float(r["KerGOPs_Median"]), "status": r["Status"],
        }
    return out


def load_his(p: Path) -> dict[tuple[int, int, int], float]:
    out = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        _, m, n, k, _g, s = line.split(";")
        out[(int(m), int(n), int(k))] = float(s) * 1e3
    return out


def gops(m: int, n: int, k: int, ms: float) -> float:
    return 2 * m * n * k / ms / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours8", nargs=3, type=Path, required=True)
    ap.add_argument("--ours1", nargs=3, type=Path, required=True)
    ap.add_argument("--his", nargs=3, type=Path, required=True)
    a = ap.parse_args()
    L8 = [load_ours(r) for r in a.ours8]
    L1 = [load_ours(r) for r in a.ours1]
    H = [load_his(p) for p in a.his]

    shapes = sorted({s for s, _ in L8[0]}, key=lambda s: int(s[1:]))
    tiles = sorted({t for _, t in L8[0]}, key=lambda t: tuple(map(int, t.split("x"))))
    dims = {s: (L8[0][(s, tiles[0])]["m"], L8[0][(s, tiles[0])]["n"], L8[0][(s, tiles[0])]["k"]) for s in shapes}

    print("## B1. Coverage and correctness status\n")
    for name, L in (("ours 8T", L8), ("ours 1T (#61)", L1)):
        for i, l in enumerate(L):
            sts = defaultdict(int)
            for v in l.values():
                sts[v["status"]] += 1
            print(f"- {name} launch {i + 1}: {len(l)} rows, {dict(sts)}")
    for i, h in enumerate(H):
        miss = [s for s in shapes if dims[s] not in h]
        print(f"- his launch {i + 1}: {len(h)} shapes in CSV, our 50 shapes missing from it: {miss or 'none'}")
    print()

    def mean_ms(L, s, t):
        return st.mean(l[(s, t)]["ms"] for l in L)

    # B2: one pass, single tile, per launch
    print("## B2. One pass over the 50 shapes, ms (GOPs by mean time)\n")
    flops = sum(2 * m * n * k for m, n, k in dims.values())
    pass_t = {t: [sum(l[(s, t)]["ms"] for s in shapes) for l in L8] for t in tiles}
    rank = sorted(tiles, key=lambda t: st.mean(pass_t[t]))
    print("| ours 8T tile | L1 | L2 | L3 | mean | GOPs |\n|---|---|---|---|---|---|")
    for t in rank[:8]:
        v = pass_t[t]
        print(f"| {t} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {st.mean(v):.1f} | {flops / st.mean(v) / 1e6:.2f} |")
    print()
    best_shape = {s: min(tiles, key=lambda t: mean_ms(L8, s, t)) for s in shapes}
    oracle = [sum(l[(s, best_shape[s])]["ms"] for s in shapes) for l in L8]
    his_pass = [sum(h[dims[s]] for s in shapes) for h in H]
    no0 = [s for s in shapes if s != "Y00"]
    flops0 = sum(2 * dims[s][0] * dims[s][1] * dims[s][2] for s in no0)
    oracle0 = [sum(l[(s, best_shape[s])]["ms"] for s in no0) for l in L8]
    his0 = [sum(h[dims[s]] for s in no0) for h in H]
    one_t = [sum(l[(s, "24x8")]["ms"] for s in shapes) for l in L1]
    o16x8 = pass_t["16x8"]
    o16x8_0 = [sum(l[(s, "16x8")]["ms"] for s in no0) for l in L8]
    print("| what | L1 | L2 | L3 | mean ms | GOPs |\n|---|---|---|---|---|---|")
    for name, v, f in (
        ("ours 1T 24x8 (#61)", one_t, flops),
        (f"ours 8T best single tile {rank[0]}", pass_t[rank[0]], flops),
        ("ours 8T best tile per shape (picked on the 3-launch mean)", oracle, flops),
        ("ours 8T at his tile 16x8", o16x8, flops),
        ("his 8T", his_pass, flops),
        ("ours 8T best per shape, without Y00", oracle0, flops0),
        ("ours 8T 16x8, without Y00", o16x8_0, flops0),
        ("his 8T, without Y00", his0, flops0),
    ):
        print(f"| {name} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {st.mean(v):.1f} | {f / st.mean(v) / 1e6:.2f} |")
    print()

    # B3: held-out check of best-per-shape (pick on one launch, score on the other two)
    print("## B3. Best tile per shape, picked on one launch, scored on the other two (ms per pass)\n")
    print("| pick on | picked tiles, scored on other launches | best single tile, same launches | gain |\n|---|---|---|---|")
    for i in range(3):
        rest = [L8[j] for j in range(3) if j != i]
        pick = {s: min(tiles, key=lambda t: L8[i][(s, t)]["ms"]) for s in shapes}
        single = min(tiles, key=lambda t: sum(L8[i][(s, t)]["ms"] for s in shapes))
        p = st.mean(sum(l[(s, pick[s])]["ms"] for s in shapes) for l in rest)
        q = st.mean(sum(l[(s, single)]["ms"] for s in shapes) for l in rest)
        print(f"| L{i + 1} | {p:.1f} | {q:.1f} ({single}) | {q - p:.1f} ms |")
    print()

    # B4: best tile per n group, margin to runner-up against launch spread
    print("## B4. Best tile per n group at 8T: margin to the runner-up against the launch spread\n")
    print("| n | shapes | best | ms/pass (L1, L2, L3) | runner-up | ms/pass | gap | best's launch range | "
          "same winner in every launch? |\n|---|---|---|---|---|---|---|---|---|")
    for n in sorted({d[1] for d in dims.values()}):
        g = [s for s in shapes if dims[s][1] == n]
        per = {t: [sum(l[(s, t)]["ms"] for s in g) for l in L8] for t in tiles}
        r = sorted(tiles, key=lambda t: st.mean(per[t]))
        b, ru = r[0], r[1]
        wins = [min(tiles, key=lambda t: per[t][i]) for i in range(3)]
        rng = max(per[b]) - min(per[b])
        print(f"| {n} | {len(g)} | {b} | {', '.join(f'{x:.2f}' for x in per[b])} | {ru} | {st.mean(per[ru]):.2f} "
              f"| {st.mean(per[ru]) - st.mean(per[b]):.2f} | {rng:.2f} | {'yes' if len(set(wins)) == 1 else 'no: ' + ', '.join(wins)} |")
    print()

    # B5: per shape table
    print("## B5. Per shape at 8T: best tile, mean GOPs and median-rep GOPs, his mean GOPs per launch\n")
    print("| shape | m x n x k | best | ours mean GOPs | ours median rep GOPs | ours 16x8 mean GOPs | his GOPs L1, L2, L3 |\n"
          "|---|---|---|---|---|---|---|")
    for s in shapes:
        m, n, k = dims[s]
        b = best_shape[s]
        g = gops(m, n, k, mean_ms(L8, s, b))
        md = st.mean(l[(s, b)]["med"] for l in L8)
        g168 = gops(m, n, k, mean_ms(L8, s, "16x8"))
        hg = ", ".join(f"{gops(m, n, k, h[dims[s]]):.2f}" for h in H)
        print(f"| {s} | {m}x{n}x{k} | {b} | {g:.2f} | {md:.2f} | {g168:.2f} | {hg} |")
    wins_mean = sum(gops(*dims[s], mean_ms(L8, s, best_shape[s])) > st.mean(gops(*dims[s], h[dims[s]]) for h in H) for s in shapes)
    wins_168 = sum(mean_ms(L8, s, "16x8") < st.mean(h[dims[s]] for h in H) for s in shapes)
    print(f"\nOurs (best tile) beats his mean on {wins_mean} of 50; ours at 16x8 beats his mean on {wins_168} of 50.")


if __name__ == "__main__":
    main()
