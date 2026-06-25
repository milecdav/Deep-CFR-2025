#!/usr/bin/env bash
# =============================================================================
# Collect job wall-clock time and memory (via seff) for each SLURM .out file in slurm_out/,
# excluding h2h-matrix outputs and the still-running job (deep-cfr-flop5-lgbm-cpu-10644569).
# Output: one line per .out file with full filename, Job wall-clock time, Memory utilized.
# Run from project root on a machine where 'seff' is available (e.g. cluster).
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_OUT="$PROJECT_ROOT/slurm_out"
OUTPUT_FILE="${1:-$PROJECT_ROOT/slurm_out/slurm_seff_report.txt}"
EXCLUDE_STILL_RUNNING="deep-cfr-flop5-lgbm-cpu-10644569.out"

cd "$PROJECT_ROOT"

# Find all .out files (relative to project root), exclude h2h-matrix and the still-running job
get_out_files() {
    find "$SLURM_OUT" -name "*.out" -type f \
        ! -path "*h2h-matrix*" \
        ! -name "$EXCLUDE_STILL_RUNNING" \
        -print | while IFS= read -r path; do
        # Path relative to project root (e.g. slurm_out/deep-cfr-flop5-10639773.out or slurm_out/old/...)
        rel="${path#$PROJECT_ROOT/}"
        echo "$rel"
    done | sort
}

# Extract job ID from filename (last number before .out)
get_job_id() {
    local base
    base=$(basename "$1" .out)
    if [[ "$base" =~ -([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

# Run seff and parse Job Wall-clock time + Memory Utilized (one line: WALLCLOCK|MEMORY)
parse_seff() {
    local jid="$1"
    local out
    out=$(seff "$jid" 2>/dev/null) || { echo "N/A|N/A"; return; }
    local wallclock memory
    wallclock=$(echo "$out" | sed -n 's/^Job Wall-clock time:[[:space:]]*//p' | head -1)
    memory=$(echo "$out" | sed -n 's/^Memory Utilized:[[:space:]]*//p' | head -1)
    echo "${wallclock:-N/A}|${memory:-N/A}"
}

echo "Scanning $SLURM_OUT for .out files (excluding h2h-matrix and $EXCLUDE_STILL_RUNNING)..."
OUT_FILES=()
while IFS= read -r f; do
    [[ -n "$f" ]] && OUT_FILES+=("$f")
done < <(get_out_files)

if [ ${#OUT_FILES[@]} -eq 0 ]; then
    echo "No files found." | tee "$OUTPUT_FILE"
    exit 0
fi

echo "Found ${#OUT_FILES[@]} file(s). Running seff for each..."
echo "Output: $OUTPUT_FILE"
echo ""

{
    echo "==============================================================================="
    echo "SLURM job report: Job wall-clock time, Memory utilized (seff)"
    echo "Generated: $(date -Iseconds 2>/dev/null || date)"
    echo "Excluded: h2h-matrix*, $EXCLUDE_STILL_RUNNING"
    echo "==============================================================================="
    echo ""
    printf "%-70s %-22s %s\n" "OUT_FILE" "JOB_WALLCLOCK" "MEMORY_UTILIZED"
    printf "%-70s %-22s %s\n" "-------" "---------------" "----------------"

    for rel in "${OUT_FILES[@]}"; do
        jid=$(get_job_id "$rel")
        [[ -z "$jid" ]] && continue
        IFS='|' read -r wallclock memory <<< "$(parse_seff "$jid")"
        printf "%-70s %-22s %s\n" "$rel" "$wallclock" "$memory"
    done
} | tee "$OUTPUT_FILE"

echo ""
echo "Done. Report written to $OUTPUT_FILE"
