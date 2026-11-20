#!/usr/bin/env bash
# Issue #128: snapshot of what the project folder produces, to prove a file move changed nothing.
#   snapshot.sh <out-dir>              take a snapshot of tests/pure_xdsl_blis_rvv as it is now
#   snapshot.sh --compare <a> <b>      print every difference; exit 1 on any
# A snapshot holds, keyed by config basename so a moved config compares with its old self:
#   generated/<config>.sha256   hash of every file the generator writes for that config
#   dryrun/<config>.txt         compile_macro_bench.sh --dry-run --config <config>, config paths cut to the basename
#   pytest.txt                  (optional, add with --pytest) passed test IDs, folder part dropped
# Runner scripts' own --dry-run plans are covered by their test_run_*.py, which run in the pytest part.
# It rewrites pj_gen/ (gitignored): never run it during a board series.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../tests/pure_xdsl_blis_rvv" && pwd)"
PY="${PY:-$PROJ/../../.venv/bin/python}"

if [ "${1:-}" = "--compare" ]; then
    [ $# -eq 3 ] || { echo "usage: $0 --compare <a> <b>" >&2; exit 2; }
    diff -r -x pytest.log "$2" "$3" && echo "snapshot compare: no difference"
    exit $?
fi

PYTEST=0
[ "${1:-}" = "--pytest" ] && { PYTEST=1; shift; }
[ $# -eq 1 ] || { echo "usage: $0 [--pytest] <out-dir> | --compare <a> <b>" >&2; exit 2; }
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
rm -rf "$OUT/generated" "$OUT/dryrun"; mkdir -p "$OUT/generated" "$OUT/dryrun"
cd "$PROJ"

# Every config of the folder, wherever it lives; run outputs and copies are not configs.
mapfile -t CONFIGS < <(find . \( -path ./logs -o -path ./pj_gen -o -path ./ref_baseline -o -path ./__pycache__ \) -prune \
    -o -type f -name 'config*.json' -print | sort)
seen=""
for cfg in "${CONFIGS[@]}"; do
    base="$(basename "$cfg" .json)"
    case " $seen " in *" $base "*) echo "two configs named $base" >&2; exit 1 ;; esac
    seen="$seen $base"
    if "$PY" generate_pure_blis_project.py --config "$cfg" > "$OUT/generated/$base.log" 2>&1; then
        (cd pj_gen && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$OUT/generated/$base.sha256"
        rm "$OUT/generated/$base.log"
    else
        # A config the generator refuses is part of the snapshot too: keep its last lines, not the full trace.
        tail -3 "$OUT/generated/$base.log" | sed "s|$PROJ/||g; s|[^ ]*/\($base\.json\)|\1|g" > "$OUT/generated/$base.failed"
        rm "$OUT/generated/$base.log"
    fi
    ./compile_macro_bench.sh --dry-run --config "$cfg" 2>&1 \
        | sed "s|[^ ]*/\($base\.json\)|\1|g" > "$OUT/dryrun/$base.txt"
done
echo "snapshot: ${#CONFIGS[@]} configs -> $OUT"

if [ "$PYTEST" = 1 ]; then
    gen_bak="$(mktemp -d)"; cp -a ref_baseline/generated/. "$gen_bak/"
    "$PY" -m pytest -q -p no:cacheprovider --maxfail=1000 -rA > "$OUT/pytest.log" 2>&1 || true
    cp -a "$gen_bak/." ref_baseline/generated/; rm -rf "$gen_bak"
    grep -E '^(PASSED|SKIPPED|XFAIL|XPASS|FAILED|ERROR) ' "$OUT/pytest.log" | sed -E 's/ - .*//; s|^([A-Z]+) ([^:]*/)?|\1 |' \
        | sort > "$OUT/pytest.txt"
    tail -1 "$OUT/pytest.log"
fi
