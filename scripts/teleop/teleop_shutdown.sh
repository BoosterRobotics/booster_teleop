#!/usr/bin/env bash
# Shut down the teleoperation stack in a controlled order.
set -uo pipefail

session=""
reason="cli"
dry_run=0
wait_s=8
wait_explicit=0
stop_failed=0
declare -A interrupted=()

die() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 2; }
info() { printf '\033[1;36m[shutdown]\033[0m %s\n' "$*"; }
warn() {
    printf '\033[1;33m[shutdown]\033[0m %s\n' "$*"
    [[ -z "${log:-}" ]] || printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$log"
}
# Bound local tool/diagnostic waits, including an unresponsive tmux server.
tmux() { timeout --kill-after=1s 2s tmux "$@"; }

while (($#)); do
    case "$1" in
        --session) [[ $# -ge 2 && -n "$2" ]] || die "--session requires a value"; session=$2; shift 2 ;;
        --reason) [[ $# -ge 2 && -n "$2" ]] || die "--reason requires a value"; reason=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --wait) [[ $# -ge 2 && -n "$2" ]] || die "--wait requires a value"; wait_s=$2; wait_explicit=1; shift 2 ;;
        -h|--help) printf 'Usage: %s --session <tmux-session> [--reason web|chord|cli] [--dry-run] [--wait seconds]\n' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ -n "$session" ]] || die "Missing --session"
[[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]] || die "Invalid tmux session name"
[[ "$wait_s" =~ ^[0-9]+$ ]] || die "--wait must be a non-negative integer number of seconds"
if [[ "$reason" == chord && $wait_explicit == 0 ]]; then
    wait_s=15  # Allow bounded deploy thread/process cleanup before session teardown.
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmux has-session -t "=$session" 2>/dev/null || die "Session does not exist: $session"
session_env="$(tmux show-options -v -t "=$session:" @booster_python_env 2>/dev/null || true)"
BOOSTER_PYTHON_ENV="${session_env:-${BOOSTER_PYTHON_ENV:-}}"
diag_dir="$(tmux show-options -v -t "=$session:" @booster_diag_dir 2>/dev/null || true)"
diag_dir="${diag_dir:-${BOOSTER_TELEMETRY_DIR:-${BOOSTER_DIAG_DIR:-$repo_root/logs/$session}}}"
report_dir="$diag_dir"
log=""
# Exactly one script executes signal/drain/analysis per run.
if [[ $dry_run != 1 ]]; then
    mkdir -p -- "$report_dir" || die "Cannot create report directory: $report_dir"
    exec {shutdown_lock}>"$diag_dir/.shutdown.lock"
    flock -n "$shutdown_lock" || { info "Shutdown already owned for this run"; exit 0; }
    log="$report_dir/shutdown-$session.log"
fi

say() { info "$*"; [[ -z "$log" ]] || printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$log"; }
phase() {
    (( dry_run )) && return 0
    PYTHONPATH="$repo_root" "${BOOSTER_PYTHON_ENV:+$BOOSTER_PYTHON_ENV/bin/}python3" -m booster_deploy.dashboard.shutdown_state "$diag_dir" "$session" "$1" "${2:-}"
}
failure_detail="shutdown interrupted"
on_exit() {
    local rc=$?
    if (( rc != 0 )); then phase failed "$failure_detail (exit $rc)" || true; fi
}
trap on_exit EXIT
phase exiting "Stopping control and input services" || die "Cannot publish shutdown phase"

pane_of() {
    tmux list-panes -s -t "=$session" -F '#{pane_id}|#{pane_title}' 2>/dev/null \
        | awk -F'|' -v want="$1" 'tolower($2) ~ want {print $1; exit}'
}

shutdown_started_ns="$("${PYTHON_BIN:-python3}" -c 'import time; print(time.monotonic_ns())')"
control_state() {
    PYTHONPATH="$repo_root" "${BOOSTER_PYTHON_ENV:+$BOOSTER_PYTHON_ENV/bin/}python3" -c '
import sys,time
from booster_deploy.telemetry.dashboard import control_state
st=control_state(sys.argv[1]); phase=sys.argv[2]; started=int(sys.argv[3])
mode,stamp=st.get("robot_mode"),st.get("robot_mode_mono_ns")
valid=type(mode) is int and mode in (0,1,2) and type(stamp) is int and 0<=time.monotonic_ns()-stamp<=(2 if phase=="initial" else 30)*1_000_000_000
if phase!="initial": valid=valid and stamp>=started and st.get("shutdown_confirmed") is True
if not valid: sys.exit(1)
print(mode)
' "$diag_dir" "${1:-initial}" "$shutdown_started_ns" 2>/dev/null
}

export_diagnostics() {
    local analyze="$repo_root/scripts/diagnostics/analyze_session.py"
    say "All producers have been requested to exit; running streaming analysis → $diag_dir"
    if [[ $dry_run == 1 ]]; then
        echo "  (dry-run) python $analyze $diag_dir"
        return
    fi
    PYTHONPATH="$repo_root" "${BOOSTER_PYTHON_ENV:+$BOOSTER_PYTHON_ENV/bin/}python3" "$analyze" "$diag_dir" >>"$log" 2>&1 \
        || { failure_detail="analysis failed; raw data retained at $diag_dir"; stop_failed=1; warn "$failure_detail"; }
    "${BOOSTER_PYTHON_ENV:+$BOOSTER_PYTHON_ENV/bin/}python3" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("file_integrity_complete") is True else 1)' "$diag_dir/analysis.json" >>"$log" 2>&1 \
        || { failure_detail="analysis file integrity incomplete; raw data retained"; stop_failed=1; }
}

interrupt_pane() {
    local pane="$1" label="$2" deadline=$((SECONDS + wait_s))
    [[ -n "$pane" ]] || return 0
    [[ -z "${interrupted[$pane]:-}" ]] || return 0
    interrupted[$pane]=1
    say "Stopping $label (Ctrl-C)"
    if [[ $dry_run == 1 ]]; then
        echo "  (dry-run) tmux send-keys -t $pane C-c"
        return 0
    fi
    tmux send-keys -t "$pane" C-c 2>/dev/null || true
    while :; do
        local cmd
        cmd="$(tmux display-message -p -t "$pane" '#{pane_current_command}' 2>/dev/null || true)"
        case "$cmd" in
            python*|"" ) ((SECONDS < deadline)) || break; sleep 0.4 ;;
            *) return 0 ;;
        esac
    done
    failure_detail="$label did not stop within ${wait_s}s; session retained"
    warn "$label did not exit within ${wait_s}s; retaining the session without destroying active producers"
    stop_failed=1
}

# Only the deploy pane can command the robot from this session.  Publisher,
# worker, recorder and dashboard panes stream or render but never publish robot
# commands, so their leftovers must not block teardown of an otherwise dead run.
live_producers() {
    local pane cmd title
    while IFS='|' read -r pane cmd title; do
        [[ -n "$pane" ]] || continue
        case "$cmd" in
            python*) ;;
            *) continue ;;
        esac
        case "${title,,}" in
            *deploy*|*control*) echo "$pane $cmd $title" ;;
        esac
    done < <(tmux list-panes -s -t "=$session" -F '#{pane_id}|#{pane_current_command}|#{pane_title}' 2>/dev/null)
}

pane_snapshot() {
    tmux list-panes -s -t "=$session" -F '#{pane_id} #{pane_current_command} #{pane_title}' 2>/dev/null | sed 's/^/    /'
}

say "Shutdown requested: session=$session reason=$reason"

session_mode="$(tmux show-options -v -t "=$session:" @booster_session_mode 2>/dev/null || true)"
if [[ "$reason" == chord ]]; then
    # Internal post-RPC cleanup path, spawned only from the successful SDK
    # response handler in the live controller. This script does not grant mode
    # permission and must not add a second cache/age/zero-ACK/ticket gate.
    say "B+X: controller completed SDK query confirmation; only exporting diagnostics and cleaning up, without rechecking the mode cache/zero-velocity ACK/ticket"
elif [[ "$session_mode" == sim ]]; then
    say "Simulation session: no real-robot mode ACK required; still waiting for all producers to drain"
else
state="$(control_state)" || {
    if [[ -n "$(live_producers)" ]]; then
        failure_detail="shutdown refused: actual mode 0/1/2 confirmation missing or stale"
        say "Shutdown refused: fresh observed mode=0(Damping)/1(PVT)/2(Walking) required; mode 3/unknown/stale is forbidden"
        say "Still running robot-commanding panes:"; live_producers | sed 's/^/    /' >&2
        exit 3
    fi
    say "No deploy/control pane is running in $session; remaining panes:"
    pane_snapshot >&2
    say "No live command publisher and no fresh robot mode; removing the inert session after flushing any recorder"
    if [[ $dry_run == 1 ]]; then
        echo "  (dry-run) tmux kill-session -t =$session"
        exit 0
    fi
    interrupt_pane "$(pane_of '^raw pico recorder$')" "raw PICO recorder"
    interrupt_pane "$(pane_of '^publisher [(]online mocap[)]$')" "Mocap publisher"
    (( stop_failed == 0 )) || exit 4
    phase closed "Inert session removed" || true
    tmux kill-session -t "=$session" || exit 5
    say "Removed inert session: $session (robot mode was not observed by this stack)"
    exit 0
}
say "Actual robot mode OK: $state"
fi

if [[ "$session_mode" != sim && "$reason" != chord ]]; then
    control_state >/dev/null || { say "Stop refused: actual mode is unknown/stale or no longer 0/1/2"; exit 3; }
fi

deploy_pane="$(tmux show-options -v -t "=$session:" @booster_deploy_pane 2>/dev/null || true)"
interrupt_pane "${deploy_pane:-$(pane_of 'deploy|mujoco simulation')}" "deploy"
# Await this shutdown's asynchronous ACK independently of pane command names.
if [[ "$session_mode" != sim && "$reason" != chord && $dry_run != 1 ]]; then
    confirmation_deadline=$((SECONDS + wait_s))
    until control_state confirmed >/dev/null; do
        if ((SECONDS >= confirmation_deadline)); then
            failure_detail="deploy confirmation deadline expired; session retained"
            say "Shutdown cannot continue: deploy has not confirmed this exit (actual mode 0/1/2; Walking also requires a zero-velocity ACK); retaining the session"
            exit 3
        fi
        sleep 0.05
    done
fi
# Ctrl-C flushes pending NPZ in the PICO recorder and BVH in the Mocap publisher.
interrupt_pane "$(pane_of '^raw pico recorder$')" "raw PICO recorder"
interrupt_pane "$(pane_of 'worker')" "retarget worker"
interrupt_pane "$(pane_of 'publisher')" "publisher"
while IFS='|' read -r pane title; do
    [[ -n "$pane" ]] || continue
    case "${title,,}" in *deploy*|*publisher*|*dashboard*) continue ;; esac
    interrupt_pane "$pane" "$title"
done < <(tmux list-panes -s -t "=$session" -F '#{pane_id}|#{pane_title}' 2>/dev/null)
if [[ "$session_mode" != sim && "$reason" != chord && $dry_run != 1 ]]; then
    control_state confirmed >/dev/null || { say "Analysis/teardown refused: exit confirmation is missing"; exit 3; }
fi
phase draining "Waiting for all registered writers to close" || die "Cannot publish drain phase"
if [[ $dry_run != 1 ]]; then
    PYTHONPATH="$repo_root" "${BOOSTER_PYTHON_ENV:+$BOOSTER_PYTHON_ENV/bin/}python3" -c '
import sys
from booster_deploy.dashboard.shutdown_state import wait_writers
wait_writers(sys.argv[1],int(sys.argv[2]))
' "$diag_dir" "$wait_s" >>"$log" 2>&1 || stop_failed=1
fi
phase analyzing "Running session analysis" || die "Cannot publish analysis phase"
export_diagnostics
if [[ $stop_failed == 1 ]]; then failure_detail="producer/writer drain or analysis incomplete; session retained"; say "Drain incomplete; analysis reflects only an incomplete snapshot; retaining tmux"; exit 4; fi

if [[ $dry_run == 1 ]]; then
    echo "  (dry-run) tmux kill-session -t =$session"
    say "Dry run complete; session was not killed"
    exit 0
fi
# Business closure is actual control/input/writer termination plus analysis,
# not script spawn or delayed kill scheduling. Dashboard HTTP is deliberately
# last; it has already permanently drained its own writer during drain phase.
phase closed "Control/input stopped; all writers drained; session analysis saved" || die "Cannot publish terminal phase"
PYTHONPATH="$repo_root" "${BOOSTER_PYTHON_ENV:+$BOOSTER_PYTHON_ENV/bin/}python3" -m booster_deploy.dashboard.shutdown_state "$diag_dir" "$session" wait-delivery
# Existing page has received explicit terminal state (or bounded 3s elapsed).
# No persistent extra server; a future refresh is allowed to fail.
say "Teleoperation is closed; closing dashboard / tmux last"
failure_detail="tmux session teardown failed"
tmux kill-session -t "=$session" || exit 5
say "Done: diagnostics and pane logs saved in $report_dir/"
