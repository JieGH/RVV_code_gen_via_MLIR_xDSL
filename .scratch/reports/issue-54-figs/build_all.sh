#!/bin/bash
# Rebuild the issue #54 brief in both languages, so the two never drift apart.
#   1. make_figs.py      every figure, English (issue-54-figs/) and Spanish (issue-54-figs/es/)
#   2. apply_keys.py     the same key-point highlights in both texts (safe to run again)
#   3. build_es.py       issue-54-brief-es.tex from the English preamble, tables and code + body_es.tex
#   4. tectonic          both PDFs
# Run from the repo root. TECTONIC=<path> if the binary is not on PATH.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
TECTONIC="${TECTONIC:-tectonic}"

"${ROOT}/.venv/bin/python" "${HERE}/make_figs.py" > /dev/null
python3 "${HERE}/apply_keys.py"
python3 "${HERE}/build_es.py"
cd "${HERE}/.."
for f in issue-54-brief.tex issue-54-brief-es.tex; do
    "${TECTONIC}" -X compile --only-cached "${f}" 2>&1 | grep -i -E "error|overfull|undefined" || true
    echo "built ${f%.tex}.pdf: $(pdfinfo "${f%.tex}.pdf" | awk '/^Pages/ {print $2}') pages"
done
