#!/bin/bash
# Shared helpers for the launchers under run_finetune_scripts/.
# Source it (don't execute it):
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/<relative path>/lib/common.sh"
#
# Preflight helpers: confirm up front that every split file exists and every
# subject it names has a preprocessed data.pt, so a broken fold is caught
# before training starts rather than partway through (e.g. fold 19 of 26).
# A subject that failed preprocessing is simply MISSING on disk, not an error.
#
# Typical use:
#   preflight_require_dir "$IMG_ROOT" "has the data been preprocessed?"
#   preflight_require_splits "$SPLIT_DIR" "split_fixed_*.txt"
#   for split_file in "$SPLIT_DIR"/split_fixed_*.txt; do
#       preflight_check_subjects "$split_file" "$IMG_ROOT"
#   done
#   preflight_finish "all split-file subjects found under $IMG_ROOT."
#
# Naming gotcha for the launchers: always write ${i} (with braces) when
# building --project_name, e.g. "..._fold-${i}". Without braces, "$i_general"
# parses as the unset variable "i_general", silently expands to nothing, and
# every fold gets the same project name. This bit a real run once.

# Count of problems found by the preflight_check_* helpers.
MISSING=0

# preflight_fail <message...> : print a preflight error and abort.
preflight_fail() {
    echo "Preflight FAILED: $*"
    exit 1
}

# preflight_require_dir <dir> [hint]
preflight_require_dir() {
    [[ -d "$1" ]] || preflight_fail "$1 does not exist${2:+ -- $2}"
}

# preflight_require_file <file> [hint]
preflight_require_file() {
    [[ -f "$1" ]] || preflight_fail "$1 not found${2:+ ($2)}"
}

# preflight_require_splits <split_dir> <glob> : at least one split file matches.
preflight_require_splits() {
    compgen -G "$1/$2" > /dev/null || preflight_fail "no split files found under $1"
}

# preflight_check_subjects <split_file> <img_root> [label]
# Reports each subject listed in the split file that lacks img_root/<subj>/data.pt
# and adds the count to MISSING.
preflight_check_subjects() {
    local split_file="$1" img_root="$2" label="${3:+ ($3)}" subj
    for subj in $(grep -vE '^(train|val|test)_subjects$' "$split_file"); do
        if [[ ! -f "$img_root/$subj/data.pt" ]]; then
            echo "  MISSING${label}: $subj (referenced in $split_file, no data.pt at $img_root/$subj/data.pt)"
            MISSING=$((MISSING + 1))
        fi
    done
}

# preflight_finish <ok_message> : abort if any problems were counted, else report OK.
preflight_finish() {
    if [[ "$MISSING" -gt 0 ]]; then
        preflight_fail "$MISSING problem(s) -- fix before running folds (see above)."
    fi
    echo "Preflight OK: $1"
    echo ""
}
