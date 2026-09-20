#!/bin/bash
# ============================================================
# gather_sweep_report.sh [HH:MM] -- collect what a scalar-pack sweep has finished so far (issue #26).
#
# With HH:MM it waits until then (today, or tomorrow if that time has passed). Then it
#   1. writes logs/sweep_scalar_<stamp>/report_<HHMM>.md (report_scalar_sweep.py): per layer, per
#      board and B3 run by run, scalar pack against the overnight vector pack;
#   2. rebuilds logs/report_2026-09-20/index.html (build_report.py), the interactive page whose
#      "Layer by layer" chart carries the scalar pack as a violet square.
# A run still in progress is not in either. Run it again when the sweep has finished.
#
# Detached, for a fixed time:
#   setsid nohup ./tests/pure_xdsl_blis_rvv/gather_sweep_report.sh 16:30 \
#       > tests/pure_xdsl_blis_rvv/logs/gather_sweep_report.out 2>&1 < /dev/null &
# ============================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

if [ -n "$1" ]; then
    [[ "$1" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || { echo "Error: HH:MM expected (got '$1')" >&2; exit 1; }
    T="$(date -d "today $1" +%s)"; NOW="$(date +%s)"
    [ "${T}" -lt "${NOW}" ] && T="$(date -d "tomorrow $1" +%s)"
    echo "[$(date +%H:%M)] waiting until $1"
    sleep $((T - NOW))
fi

echo "[$(date +%H:%M)] gathering"
"${PY}" "${HERE}/report_scalar_sweep.py" > /dev/null && echo "  report written"
"${PY}" "${HERE}/logs/report_2026-09-20/build_report.py" && echo "  page rebuilt"
echo "[$(date +%H:%M)] done"
