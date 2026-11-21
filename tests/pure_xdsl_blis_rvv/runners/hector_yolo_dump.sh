# hector_yolo_dump.sh -- sourced by the YOLO runners (user, 2026-11-16): every design run on Héctor's YOLO test
# files its report and logs in .scratch/reports/issue-<N>/<series folder name>/, after every launch and at the end.
#
# hector_yolo_dump <issue> <series dir> -- copies the series folder there (refreshing an earlier partial copy)
HECTOR_YOLO_DIR="${HECTOR_YOLO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.scratch/reports}"  # env: tests; one place for all reports (user, 2026-11-17)
hector_yolo_dump() {
    local dest="${HECTOR_YOLO_DIR}/issue-$1/$(basename "$2")"
    mkdir -p "${dest}" && cp -r "$2"/. "${dest}"/ && echo "Report copy: ${dest}"
}
