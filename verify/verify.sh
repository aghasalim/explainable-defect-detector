#!/usr/bin/env bash
# Recompute every published number from the rawer file it came from, in a
# language that is not the one that produced it.
#
# Everything this repository reports came out of one Python process. The metrics
# in reports/*.json were computed by the script that wrote them, the tables in
# reports/*.md were formatted by the script that wrote those, and the sentences
# in README.md and notes/METHODS.md were typed out of both. Nothing ever read
# any of it back. That is how a stale model size and a control of 0.01 for a
# category whose control is 0.00 both survived in the README.
#
# Each check below starts from the rawest file that still exists and rebuilds
# what is published from it. A mistake now has to be made identically in two
# languages to go unnoticed.
#
# Every check is skipped with a clear message if its toolchain is absent, so
# this runs on a laptop with only some of them. CI has all of them.
set -uo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0
tmp="${TMPDIR:-/tmp}"

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then
        printf -- '--- %s: passed\n' "$name"; pass=$((pass + 1))
    else
        printf -- '--- %s: FAILED\n' "$name"; fail=$((fail + 1))
    fi
}

# The SQL scripts assert with instr() and print ok or FAIL per line, since
# sqlite3 has no way to exit non-zero on a failed comparison.
check_sql () {
    local out
    out=$(sqlite3 -init "verify/$1" :memory: "" 2>&1)
    printf '%s\n' "$out"
    if printf '%s\n' "$out" | grep -q '^FAIL'; then
        return 1
    fi
    if ! printf '%s\n' "$out" | grep -q '^rebuilt '; then
        echo "the script did not reach its summary line"
        return 1
    fi
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o "$tmp/auroc" verify/auroc.c -lm || return 1
    "$tmp/auroc" reports/baseline_*.json reports/patchcore*.json
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () {
    ( cd verify/tolerance && cargo run --release --quiet -- "$root" \
        "$root"/reports/baseline_*.json "$root"/reports/patchcore*.json )
}

run "SQL, prose and section 3 tables" sqlite3 check_sql aggregate.sql
run "SQL, the EDA report"             sqlite3 check_sql eda.sql
run "C, detection metrics"            cc      check_c
run "Go, structure and counts"        go      check_go
run "R, statistical inference"        Rscript Rscript verify/verify.R "$root"
run "Rust, exhaustive AUROC and coverage" cargo check_rust
run "JavaScript, published tables"    node    node verify/tables.mjs "$root"
run "Java, calibration tables"        java    java verify/Calibration.java "$root"
run "Ruby, held-out split"            ruby    ruby verify/splits.rb "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
