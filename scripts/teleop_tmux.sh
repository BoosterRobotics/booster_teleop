#!/usr/bin/env bash
#
# Single teleoperation entry point for live or recorded input, using the fixed
# M0-1010 retargeted policy in simulation or on the real robot.
#
# Selections other than the automatic offline clip have prompts and CLI flags; the script
# can be used with zero flags the first time and as one line once the operator
# knows what they want.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/teleop_tmux.sh
  bash scripts/teleop_tmux.sh host pico|mocap [sim] [options]
  bash scripts/teleop_tmux.sh onboard pico|mocap sim|real [options]

Positional (optional, prompted when omitted):
  host|onboard            Deployment machine
  pico|mocap              Input source
  sim|real                Backend (host supports sim only)

Options:
  --task t2_gmc_rt            Policy task (only t2_gmc_rt is supported)
  --replay online|offline    Live input, or replay a recorded clip
  --recording PATH           Explicit offline clip (highest priority; no path prompt).
                             Omitted: newest compatible clip by mtime in logs/pico_raw
                             (raw PICO NPZ) or logs/mocap_raw (PN-Link BVH), for the chosen
                             source. Invalid newest clips are skipped; none -> error.
                             No recording-path environment default overrides this CLI.
                             Change the clip later in the offline dashboard.
  --record on|off            Enable live trajectory recording (PICO or onboard Noitom)
  --record-dir DIR           Clip output directory (default: logs/pico_raw or logs/mocap_raw)
  --hand-control MODE        Hand control: on or off (default: on); also BOOSTER_HAND_CONTROL
  --hands MODE               Alias for --hand-control
  --no-hands                 Alias for --hand-control off
  --session-name NAME        Override the tmux session name
  --SN NUMBER                Robot serial number stored in this run (digits only; optional)
  --env-path PATH            Override the Python environment
  --no-attach                Create the tmux session without attaching
  --no-diag                  Disable console [diag] lines only (default: on).
                             Telemetry uses booster_deploy/telemetry/telemetry.yaml;
                             default run directory: logs/<session>/
  --close-ui                 Disable the dashboard and use the terminal/tmux path.
                             Dashboard default: on, session detached, port 8095.
  --dashboard                Enable the dashboard explicitly (compatibility flag).
  --lang zh|en               UI language (browser preference, then Chinese; no prompt)
  --dashboard-port PORT      Dashboard port (default: 8095)
  --device-host IP           Motion device IP (optional; overrides PICO peer discovery)
  --dashboard-bind ADDR      Dashboard bind address (default: 0.0.0.0)
  --dashboard-keyboard-overlay    Accept dashboard A/Y/B controls in PICO mode (default)
  --no-dashboard-keyboard-overlay Disable this PICO keyboard overlay
  --dry-run                  Print the pane commands without starting or stopping programs
  -h, --help                 Show this help.

Offline replay on the real robot: press the robot controller's A once to replay
one cycle from the first frame. At the end, the robot uses the same smooth
return to Walking as B. Press A again for the next cycle; B ends it early.

Teleop raw recording: hold both triggers and press X to start, Y to stop and
save. Clips are written as raw PICO NPZ or Noitom BVH files for offline replay.
When enabled, both sources use the Data recording row in the dashboard.

The Python environment defaults to BOOSTER_PYTHON_ENV, the active venv, then
~/.venvs/booster_teleoperation.
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

prompt() {
    local label=$1 default_value=$2 answer
    printf '%s [%s]: ' "$label" "$default_value" >&2
    IFS= read -r answer || die "selection is required; pass the values as arguments for non-interactive use"
    printf '%s' "${answer:-$default_value}"
}

# ------------------------------------------------------------------ defaults
role=""
source_kind=""
run_mode=""
replay=""
recording_path=""
readonly task=t2_gmc_rt input_path=retarget
hand_control="${BOOSTER_HAND_CONTROL:-}"
record=""
record_dir=""
session_name=""
sn=""
default_env_path="${BOOSTER_PYTHON_ENV:-${VIRTUAL_ENV:-${HOME}/.venvs/booster_teleoperation}}"
env_path=$default_env_path
attach=1
dry_run=0
diag=1
dashboard=1
dashboard_port=8095
device_host=""
dashboard_bind=0.0.0.0
dashboard_lang="" # Preserve browser preference unless the language is explicit.
dashboard_overlay=1

# ------------------------------------------------------------------ arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            [[ $# -ge 2 && -n "$2" && "$2" != '--'* ]] || die "--task requires a value"
            [[ "$2" == "$task" ]] || die "Unsupported --task: $2; expected $task"
            shift 2
            ;;
        --replay|--recording|--record|--record-dir|--hand-control|--hands)
            [[ $# -ge 2 && -n "$2" && "$2" != '--'* ]] || die "$1 requires a value"
            case "$1" in
                --replay) replay=$2 ;;
                --recording) recording_path=$2 ;;
                --record) record=$2 ;;
                --record-dir) record_dir=$2 ;;
                --hand-control|--hands) hand_control=$2 ;;
            esac
            shift 2
            ;;
        --session-name|--SN|--env-path|--dashboard-port|--dashboard-bind|--device-host)
            [[ $# -ge 2 && -n "$2" && "$2" != '--'* ]] || die "$1 requires a value"
            case "$1" in
                --session-name) session_name=$2 ;;
                --SN) sn=$2 ;;
                --env-path) env_path=$2 ;;
                --dashboard-port) dashboard_port=$2 ;;
                --device-host) device_host=$2 ;;
                --dashboard-bind) dashboard_bind=$2 ;;
            esac
            shift 2
            ;;
        --lang)
            [[ $# -ge 2 ]] || die "--lang requires zh|en"
            case "$2" in zh|en) dashboard_lang=$2 ;; *) die "Invalid --lang: $2; expected zh|en" ;; esac
            shift 2 ;;
        --no-hands) hand_control=off; shift ;;
        --dashboard) dashboard=1; shift ;;
        --close-ui) dashboard=0; shift ;;
        --dashboard-keyboard-overlay) dashboard_overlay=1; shift ;;
        --no-dashboard-keyboard-overlay) dashboard_overlay=0; shift ;;
        --no-attach) attach=0; shift ;;
        --no-diag) diag=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) die "Unknown option: $1" ;;
        *)
            if [[ -z "$role" ]]; then role=$1
            elif [[ -z "$source_kind" ]]; then source_kind=$1
            elif [[ -z "$run_mode" ]]; then run_mode=$1
            else die "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

# ------------------------------------------------------------------ output styling
# Colour only on a real terminal (and honour NO_COLOR), so logs stay plain.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    c_reset=$'\033[0m'
    c_url=$'\033[1;36m'
    c_dim=$'\033[2m'
    c_ok=$'\033[1;32m'
    c_warn=$'\033[1;33m'
else
    c_reset=''
    c_url=''
    c_dim=''
    c_ok=''
    c_warn=''
fi

# Print a reusable command for the selected options.
print_reuse() {
    local reuse="bash scripts/teleop_tmux.sh $role $source_kind $run_mode --replay $replay"
    reuse+=" --task $task"
    reuse+=" --record $record --hand-control $hand_control"
    if [[ "$replay" == offline ]]; then printf -v reuse '%s --recording %q' "$reuse" "$recording_path"; fi
    if [[ "$record" == on && "$(realpath -m -- "$record_dir")" != "$(realpath -m -- "$default_record_dir")" ]]; then
        printf -v reuse '%s --record-dir %q' "$reuse" "$record_dir"
    fi
    if (( dashboard == 0 )); then reuse+=" --close-ui"; fi
    if (( attach == 0 && dashboard == 0 )); then reuse+=" --no-attach"; fi
    if (( diag == 0 )); then reuse+=" --no-diag"; fi
    if [[ ! "$BOOSTER_PYTHON_ENV" -ef "$default_env_path" ]]; then
        printf -v reuse '%s --env-path %q' "$reuse" "$BOOSTER_PYTHON_ENV"
    fi
    if [[ "$dashboard_port" != 8095 ]]; then printf -v reuse '%s --dashboard-port %q' "$reuse" "$dashboard_port"; fi
    if [[ "$dashboard_bind" != 0.0.0.0 ]]; then printf -v reuse '%s --dashboard-bind %q' "$reuse" "$dashboard_bind"; fi
    if [[ -n "$sn" ]]; then reuse+=" --SN $sn"; fi
    if [[ -n "$device_host" ]]; then printf -v reuse '%s --device-host %q' "$reuse" "$device_host"; fi
    if [[ -n "$dashboard_lang" ]]; then reuse+=" --lang $dashboard_lang"; fi
    if (( dashboard_overlay )); then reuse+=" --dashboard-keyboard-overlay"; else reuse+=" --no-dashboard-keyboard-overlay"; fi
    printf '\n%sReuse these options (copy and run):%s\n  %s%s%s\n' "$c_dim" "$c_reset" "$c_url" "$reuse" "$c_reset"
}

# Addresses a browser on another machine can use for the dashboard. The dashboard
# binds 0.0.0.0, so a same-network client needs no tunnel at all; prefer the
# address this ssh session already reached (SSH_CONNECTION) over listing every NIC.
dashboard_addrs() {
    local found=()
    if [[ -n "${SSH_CONNECTION:-}" ]]; then
        local client_ip server_ip
        read -r client_ip _ server_ip _ <<<"$SSH_CONNECTION"
        # The address the operator's ssh already reached is the best candidate;
        # skip loopback, which is what a relayed connection reports.
        if [[ -n "$server_ip" && "$server_ip" != 127.* ]]; then
            found+=("$server_ip")
        fi
    fi
    local ip
    while IFS= read -r ip; do
        [[ -n "$ip" ]] && found+=("$ip")
    done < <(ip -4 -o addr show scope global 2>/dev/null \
        | awk '$2 !~ /^(lo|docker|br-|virbr|veth|usb)/ {print $4}' \
        | cut -d/ -f1)
    ((${#found[@]})) || return 0
    printf '%s\n' "${found[@]}" | awk '!seen[$0]++'
}

wait_dashboard() {
    local port=$1 tries=${2:-25} i=0
    while ((i < tries)); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

# ------------------------------------------------------------------ interactive selection
# Steps are numbered as they are actually asked; host runs skip the mode question.
step_no=0
step_header() {
    step_no=$((step_no + 1))
    printf 'Step %d: %s\n' "$step_no" "$*" >&2
}

if [[ -z "$role" ]]; then
    step_header "choose deployment machine"
    echo "  1) Host computer (Linux x86_64)" >&2
    echo "  2) Onboard computer (Linux aarch64)" >&2
    default_role=1
    [[ "$(uname -m)" != aarch64 && "$(uname -m)" != arm64 ]] || default_role=2
    role=$(prompt "Enter 1 for Host or 2 for Onboard" "$default_role")
fi
case "$role" in
    1) role=host ;; 2) role=onboard ;;
    host|onboard) ;;
    *) die "role must be host or onboard" ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "$role" == onboard ]] && (( dry_run == 0 )); then
    source "$repo_root/scripts/check_booster_version.sh"
    require_booster_version || exit 1
fi

if [[ -z "$source_kind" ]]; then
    step_header "choose input source"
    echo "  1) Mocap" >&2
    echo "  2) Pico" >&2
    source_kind=$(prompt "Enter 1 for Mocap or 2 for Pico" "1")
fi
case "$source_kind" in
    1) source_kind=mocap ;; 2) source_kind=pico ;;
    mocap|pico) ;;
    *) die "input source must be pico or mocap" ;;
esac

if [[ -z "$run_mode" ]]; then
    if [[ "$role" == host ]]; then
        run_mode=sim
    else
        step_header "choose deployment mode"
        echo "  1) Simulation" >&2
        echo "  2) Real robot" >&2
        run_mode=$(prompt "Enter 1 for Sim or 2 for Real" "1")
    fi
fi
case "$run_mode" in
    1) run_mode=sim ;; 2) run_mode=real ;;
    sim|real) ;;
    *) die "deployment mode must be sim or real" ;;
esac
[[ "$role" != host || "$run_mode" == sim ]] || die "Host supports sim only; use onboard for real-robot deployment"

# A closed stdin (CI, tests, scripted runs) must never block on a menu.
interactive=1
if [[ ! -t 0 ]]; then
    interactive=0
fi
if [[ -z "$replay" ]]; then
    (( interactive )) || replay=online
fi

if [[ -z "$replay" ]]; then
    step_header "choose input timeline"
    echo "  1) Online  (live input from the source device)" >&2
    echo "  2) Offline (replay a recorded clip)" >&2
    replay=$(prompt "Enter 1 for Online or 2 for Offline" "1")
fi
case "$replay" in
    1) replay=online ;; 2) replay=offline ;;
    online|offline) ;;
    *) die "replay must be online or offline" ;;
esac

# ------------------------------------------------------------------ Python environment
default_env_path=${default_env_path/#\~/$HOME}
if [[ "$default_env_path" != /* ]]; then default_env_path="$PWD/$default_env_path"; fi
env_path=${env_path/#\~/$HOME}
if [[ "$env_path" != /* ]]; then env_path="$PWD/$env_path"; fi
[[ -f "$env_path/bin/activate" && -x "$env_path/bin/python" ]] \
    || die "Python environment not found: $env_path; run bash scripts/install_env.sh or pass --env-path"
cd "$repo_root"
BOOSTER_PYTHON_ENV=$(cd "$env_path" && pwd)
export BOOSTER_PYTHON_ENV
python_bin="$BOOSTER_PYTHON_ENV/bin/python"

# No trajectory menu/input at startup. CLI wins; otherwise choose the newest
# compatible clip for the selected source, never infer or change that source.
if [[ "$replay" == offline && -z "$recording_path" ]]; then
    recording_path=$(PYTHONPATH="$repo_root" "$python_bin" -m booster_deploy.inputs.recording_selection \
        --repo-root "$repo_root" --source "$source_kind") || die "No compatible default offline recording is available"
    printf 'Replaying latest compatible recording: %s\n' "$recording_path" >&2
fi

# Noitom recording uses the robot ROS remote-controller topic (onboard only).
record_prompt_applies=0
if [[ "$replay" == online && ( "$source_kind" == pico || "$role" == onboard ) ]]; then
    record_prompt_applies=1
fi
if [[ -z "$record" ]]; then
    if (( interactive )) && (( record_prompt_applies )); then
        step_header "traj recording for this session"
        echo "  1) Off" >&2
        echo "  2) On  (see Data Recording and Replay in the README)" >&2
        record=$(prompt "Enter 1 to keep recording off or 2 to enable it" "1")
    else
        record=off
    fi
fi
case "$record" in
    1|off|no|false|0) record=off ;;
    2|on|yes|true) record=on ;;
    *) die "record must be on or off" ;;
esac

# Mocap can retarget hands in simulation and replay as well as on the real robot.
hand_prompt_applies=0
if [[ "$source_kind" == mocap || ( "$replay" == online && "$run_mode" == real ) ]]; then
    hand_prompt_applies=1
fi
if [[ -z "$hand_control" ]]; then
    if [[ -t 0 ]] && (( hand_prompt_applies )); then
        step_header "control the real robot's dexterous hands?"
        echo "  1) On" >&2
        echo "  2) Off" >&2
        hand_control=$(prompt "Enter 1 or 2" "1")
    else
        hand_control=on
    fi
fi
case "$hand_control" in
    1|on|yes|true) hand_control=on ;;
    2|off|no|false) hand_control=off ;;
    *) die "hand control must be on or off" ;;
esac

# ------------------------------------------------------------------ consistency checks
if [[ "$replay" == offline ]]; then
    [[ -n "$recording_path" ]] || die "offline replay needs a recording path (--recording)"
    case "$recording_path" in
        "~") recording_path=$HOME ;;
        "~/"*) recording_path="$HOME/${recording_path#\~/}" ;;
    esac
    [[ -f "$recording_path" ]] || die "recording not found on this machine: $recording_path"
    recording_path=$(realpath -- "$recording_path")
    if [[ "$source_kind" == mocap ]]; then
        [[ "${recording_path,,}" == *.bvh ]] || die "mocap offline replay needs a .bvh recording: $recording_path"
    else
        [[ "${recording_path,,}" == *.npz ]] || die "pico offline replay needs a .npz recording: $recording_path"
    fi
elif [[ -n "$recording_path" ]]; then
    die "--recording requires --replay offline"
fi
pico_record=0
if [[ "$record" == on ]]; then
    [[ "$replay" == online ]] || die "trajectory recording requires online input"
    (( record_prompt_applies )) || die "Noitom trajectory recording requires the onboard robot remote-controller topic"
    if [[ "$source_kind" == pico ]]; then pico_record=1; fi
fi
default_record_dir=$repo_root/logs/${source_kind}_raw
record_dir=${record_dir:-$default_record_dir}
record_dir=${record_dir/#\~/$HOME}

session_timestamp=${session_timestamp:-$(date +%Y%m%d-%H%M%S)}
if [[ -z "$session_name" ]]; then
    session_name="booster-teleop-${role}-${source_kind}-${run_mode}-${replay}-${session_timestamp}"
fi
[[ "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]] || die "session name may contain only letters, digits, dots, underscores, or hyphens"
[[ -z "$sn" || "$sn" =~ ^[0-9]+$ ]] || die "--SN must contain digits only"
[[ "$dashboard_port" =~ ^[0-9]{1,5}$ ]] && (( 10#$dashboard_port >= 1 && 10#$dashboard_port <= 65535 )) \
    || die "--dashboard-port must be an integer between 1 and 65535"

diag_dir=${BOOSTER_TELEMETRY_DIR:-${BOOSTER_DIAG_DIR:-$repo_root/logs/$session_name}}
BOOSTER_RUN_ORIGIN_NS=$("$python_bin" -c 'import time; print(time.monotonic_ns())')
telemetry_config="$repo_root/booster_deploy/telemetry/telemetry.yaml"
telemetry_topology=retarget
telemetry_mjcf=$repo_root/booster_deploy/assets/t2/t2_31dof_fc.xml
if [[ "$hand_control" == on ]]; then
    telemetry_mjcf=$repo_root/booster_deploy/assets/t2/t2_31dof_revo2hand.xml
fi
if [[ "$replay" == offline ]]; then
    input_file="${recording_path##*/}"
    trajectory_path="$recording_path"
    trajectory_stem="$(basename "${recording_path%.*}")"
else
    input_file=online
    trajectory_path=""
    trajectory_stem=""
fi
# Never append a new invocation to an existing run, even with a reused session name.
if [[ -e "$diag_dir" ]]; then
    diag_dir="${diag_dir}.run-$(date +%Y%m%d-%H%M%S)-$$"
fi
# The dashboard owns the only externally reachable port; when it is enabled the
# viser viewers are bound to loopback so a second port never has to be opened.
if (( dashboard )); then
    attach=0
    viser_bind=127.0.0.1
else
    viser_bind=0.0.0.0
fi
# A detached session has no client, so tmux falls back to `default-size` (80x24 on
# a stock image) and the 6th/7th pane cannot be split.  Give it a real size; a
# client attaching later simply resizes it.
session_size_args=()
if (( attach == 0 )); then
    session_size_args=(-x 240 -y 60)
fi
env_script="scripts/env_set.sh"
if (( dry_run == 0 )); then
    [[ "$(uname -s)" == Linux ]] || die "Linux is required"
    if [[ "$role" == host ]]; then
        [[ "$(uname -m)" == x86_64 ]] || die "Host requires Linux x86_64"
    else
        [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]] || die "Onboard requires Linux AArch64"
    fi
    if [[ "$replay" == offline ]]; then
        source "$repo_root/$env_script" "$role" offline
    else
        source "$repo_root/$env_script" "$role" "$source_kind" \
            "${BOOSTER_MOCAP_LOCAL_IP:-}" "${BOOSTER_MOCAP_SERVER_IP:-}"
    fi
fi
if [[ "$source_kind" == pico ]]; then
    hand_trigger_control="${BOOSTER_HAND_TRIGGER_CONTROL:-1}"
    hand_retarget_control="${BOOSTER_HAND_RETARGET_CONTROL:-0}"
elif [[ "$replay" == offline ]]; then
    # env_set.sh disables live inputs in offline mode; the BVH hand worker is local.
    hand_trigger_control=0
    hand_retarget_control=1
else
    hand_trigger_control="${BOOSTER_HAND_TRIGGER_CONTROL:-0}"
    hand_retarget_control="${BOOSTER_HAND_RETARGET_CONTROL:-1}"
fi

# ------------------------------------------------------------------ pane commands
if [[ "$replay" == offline ]]; then
    publisher="scripts/diagnostics/offline_${source_kind}_publisher.py"
else
    publisher="scripts/teleop/${source_kind}_rt_zmq_publisher.py"
fi
worker="scripts/teleop/retarget_rt_zmq_worker.py"
hand_worker="scripts/teleop/hand_retarget_rt_zmq_worker.py"
robot_viewer="scripts/teleop/retarget_rt_zmq_robot_viewer.py"
human_viewer="scripts/teleop/${source_kind}_rt_zmq_human_viewer.py"
recorder="scripts/diagnostics/record_pico_stream.py"
worker_source="${source_kind}_rt"

entrypoints=("$env_script" "$publisher" "$worker" "scripts/deploy.py")
(( dashboard )) || entrypoints+=("$human_viewer" "$robot_viewer")
[[ "$source_kind" != mocap || "$hand_control" != on ]] || entrypoints+=("$hand_worker")
(( ! pico_record )) || entrypoints+=("$recorder")
(( dashboard == 0 )) || entrypoints+=("scripts/dashboard.py")
for entrypoint in "${entrypoints[@]}"; do
    [[ -f "$repo_root/$entrypoint" ]] || die "Entry point not found: $repo_root/$entrypoint"
done

pane_command() {
    local command=$1 setup
    if [[ "$replay" == offline ]]; then
        printf -v setup 'export BOOSTER_PYTHON_ENV=%q; source %q %q offline' \
            "$BOOSTER_PYTHON_ENV" "$repo_root/$env_script" "$role"
    else
        printf -v setup 'export BOOSTER_PYTHON_ENV=%q; source %q %q %q' \
            "$BOOSTER_PYTHON_ENV" "$repo_root/$env_script" "$role" "$source_kind"
    fi
    if [[ "$source_kind" == mocap && "$replay" != offline ]]; then
        printf -v setup '%s %q %q' "$setup" "${BOOSTER_MOCAP_LOCAL_IP:-192.168.10.102}" "${BOOSTER_MOCAP_SERVER_IP:-192.168.10.199}"
    fi
    printf -v setup '%s && export BOOSTER_DIAG=%q BOOSTER_DIAG_DIR=%q BOOSTER_RUN_ID=%q BOOSTER_RUN_ORIGIN_NS=%q BOOSTER_TELEMETRY_DIR=%q BOOSTER_TELEMETRY_CONFIG=%q BOOSTER_TELEMETRY_TOPOLOGY=%q BOOSTER_TELEMETRY_EXECUTION=%q BOOSTER_TELEMETRY_TASK=%q BOOSTER_TELEMETRY_INPUT_SOURCE=%q BOOSTER_TELEMETRY_MJCF=%q BOOSTER_TELEMETRY_INPUT_FILE=%q BOOSTER_TELEMETRY_INPUT_PATH=%q BOOSTER_TELEMETRY_REPLAY=%q BOOSTER_TELEMETRY_SN=%q BOOSTER_TELEMETRY_TRAJECTORY_PATH=%q BOOSTER_TELEMETRY_TRAJECTORY_STEM=%q' "$setup" "$diag" "$diag_dir" "$session_name" "$BOOSTER_RUN_ORIGIN_NS" "$diag_dir" "$telemetry_config" "$telemetry_topology" "$run_mode" "$task" "$source_kind" "$telemetry_mjcf" "$input_file" "$input_path" "$replay" "$sn" "$trajectory_path" "$trajectory_stem"
    # Lets the deploy pane trigger the ordered shutdown script (PICO B+X long press)
    # without hardcoding the session name anywhere.
    printf -v setup '%s BOOSTER_TMUX_SESSION=%q' "$setup" "$session_name"
    if (( dashboard )) && (( dashboard_overlay )); then
        # Lets the deploy pane accept the dashboard's r/y/b keystrokes in PICO mode.
        printf -v setup '%s && export BOOSTER_CONTROL_KEYBOARD_OVERLAY=1' "$setup"
    fi
    if [[ "$hand_control" == on ]]; then
        printf -v setup '%s && export BOOSTER_HAND_CONTROL_ENABLED=1 BOOSTER_HAND_CONTROL_MODE=environment BOOSTER_HAND_TRIGGER_CONTROL=%q BOOSTER_HAND_RETARGET_CONTROL=%q' \
            "$setup" "$hand_trigger_control" "$hand_retarget_control"
    else
        printf -v setup '%s && export BOOSTER_HAND_CONTROL_ENABLED=0 BOOSTER_HAND_CONTROL_MODE=environment BOOSTER_HAND_TRIGGER_CONTROL=0 BOOSTER_HAND_RETARGET_CONTROL=0' "$setup"
    fi
    if [[ "$replay" == offline ]]; then
        # Offline ports: publisher 5555 (mocap 5557) -> worker 5556 -> deploy.
        # Keep these aligned with the worker flags above and with the viewer hosts.
        setup+=" && export BOOSTER_RT_REFERENCE_HOST=127.0.0.1 BOOSTER_RT_REFERENCE_PORT=5556 BOOSTER_PICO_CONTROL_HOST=127.0.0.1 BOOSTER_PICO_CONTROL_PORT=5555 BOOSTER_RT_RESET_HOST=127.0.0.1 BOOSTER_RT_RESET_PORT=$rt_reset_port BOOSTER_HAND_RETARGET_HOST=127.0.0.1"
        if [[ "$run_mode" == real ]]; then
            # The replay publisher turns the clip end into "leave Tracking", so the
            # deploy must be told it is driving a replay, not live input.
            setup+=' && export BOOSTER_OFFLINE_REPLAY=1 BOOSTER_REMOTE_CONTROL_BACKEND=joystick-robot'
        fi
    fi
    printf '%s && %s' "$setup" "$command"
}

# publisher
if [[ "$replay" == offline ]]; then
    if [[ "$source_kind" == mocap ]]; then
        printf -v publisher_cmd 'python %q %q --bind 127.0.0.1 --port 5557' "$repo_root/$publisher" "$recording_path"
        [[ "$run_mode" != real ]] || publisher_cmd+=" --no-loop --wait-for-start-button"
    else
        printf -v publisher_cmd 'python %q %q --bind 127.0.0.1 --port 5555' "$repo_root/$publisher" "$recording_path"
        if [[ "$run_mode" == real ]]; then
            # One cycle per A press, aborted by B; the clip end returns to Walking.
            publisher_cmd+=" --no-loop --wait-for-start-button"
        else
            publisher_cmd+=" --loop"
        fi
    fi
else
    printf -v publisher_cmd 'python %q' "$repo_root/$publisher"
fi
# The recorder pane watches pico_rt.raw, so the publisher must mask the trigger+X/Y
# chords itself -- without this the bare face button reaches the robot state machine.
if (( pico_record )); then
    publisher_cmd+=" --control-chord-mask --raw-bind tcp://127.0.0.1:5581"
elif [[ "$record" == on ]]; then
    printf -v publisher_cmd '%s --record-dir %q' "$publisher_cmd" "$record_dir"
fi

# retarget worker
printf -v worker_cmd 'python %q --input-source %q' "$repo_root/$worker" "$worker_source"
# Publish the actual worker reset port in tmux metadata for the dashboard.
rt_reset_port=${BOOSTER_RT_RESET_PORT:-5559}
if [[ "$replay" == offline ]]; then
    offline_source_port=5555
    [[ "$source_kind" != mocap ]] || offline_source_port=5557
    worker_cmd+=" --input-port $offline_source_port --output-bind 127.0.0.1 --output-port 5556 --reset-bind 127.0.0.1 --reset-port 5559 --robot-name t2 --include-human-frame"
    rt_reset_port=5559
else
    if [[ "$run_mode" == sim ]]; then
        worker_cmd+=" --robot-name t2 --include-human-frame"
    fi
fi
if [[ "$run_mode" == sim && "$source_kind" == mocap && "$hand_control" == on ]]; then
    worker_cmd+=" --include-hand-frame"
fi
printf -v hand_worker_cmd 'python %q' "$repo_root/$hand_worker"

# viewers
# Robot front view, locked: azimuth 180° is this project's "front" convention (see
# tests/trajectory_comparison), and --follow-robot keeps pushing the root-centred pose,
# so a stray drag snaps back instead of losing the view mid-teleoperation.
robot_camera_args="--camera-azimuth 180 --camera-elevation -10 --camera-distance 2.4 --follow-robot"
# The human viewers have no follow switch; same front direction, auto-fit distance.
human_camera_args="--camera-azimuth 180 --camera-elevation -10"
if [[ "$replay" == offline ]]; then
    printf -v viewer_cmd 'python %q --host 127.0.0.1 --port 5556 --robot-name t2 --viser-host %s --viser-port 8081 %s' "$repo_root/$robot_viewer" "$viser_bind" "$robot_camera_args"
    printf -v human_viewer_cmd 'python %q --host 127.0.0.1 --port %s --viser --viser-host %s --viser-port 8080 %s' \
        "$repo_root/$human_viewer" "$([[ "$source_kind" == mocap ]] && echo 5557 || echo 5555)" "$viser_bind" "$human_camera_args"
else
    printf -v viewer_cmd 'python %q --viser-host %s --viser-port 8081 %s' "$repo_root/$robot_viewer" "$viser_bind" "$robot_camera_args"
    printf -v human_viewer_cmd 'python %q --viser --viser-host %s --viser-port 8080 %s' "$repo_root/$human_viewer" "$viser_bind" "$human_camera_args"
fi
[[ "$hand_control" != off ]] || viewer_cmd+=" --no-with-hands"

# deploy
printf -v deploy_cmd 'python %q' "$repo_root/scripts/deploy.py"
if [[ "$source_kind" == mocap && "$record" == on ]]; then
    deploy_cmd="BOOSTER_MOCAP_RECORDING=1 $deploy_cmd"
fi
if (( dashboard )); then
    # Sim and real controllers publish latest-only robot state to the dashboard
    # component; this is local IPC and never blocks the control loop.
    deploy_cmd="BOOSTER_WEB_VIS_ENDPOINT=tcp://127.0.0.1:5566 $deploy_cmd"
fi
if [[ "$run_mode" == sim ]]; then
    deploy_cmd+=" --mujoco"
    if (( dashboard == 0 )); then
        deploy_cmd+=" --viser --viser-host $viser_bind --viser-port 8082"
        deploy_title="MuJoCo simulation :8082"
    else
        deploy_title="MuJoCo simulation → webviz"
    fi
else
    deploy_cmd+=" --real"
    deploy_title="real robot deploy"
fi

# recorder
if (( pico_record )); then
    printf -v record_cmd 'python %q --output-dir %q --endpoint tcp://127.0.0.1:5581' \
        "$repo_root/$recorder" "$record_dir"
    # Dashboard mode controls recording with 'x'/'y' keystrokes (same code path as
    # the trigger chords); terminal mode keeps the chords only.
    (( dashboard == 0 )) || record_cmd+=" --keyboard-control"
fi

# Dashboard: the web UI and its APIs share one port.
if (( dashboard )); then
    printf -v dashboard_cmd 'python %q --session %q --bind %q --port %q --repo-root %q' \
        "$repo_root/scripts/dashboard.py" "$session_name" "$dashboard_bind" "$dashboard_port" "$repo_root"
    if [[ -n "$device_host" ]]; then
        printf -v dashboard_cmd '%s --device-host %q' "$dashboard_cmd" "$device_host"
    fi
    if [[ -n "$dashboard_lang" ]]; then
        printf -v dashboard_cmd '%s --lang %q' "$dashboard_cmd" "$dashboard_lang"
    fi
    if (( diag )); then
        printf -v dashboard_cmd '%s --diag-dir %q' "$dashboard_cmd" "$diag_dir"
    fi
fi


# ------------------------------------------------------------------ dry run
if (( dry_run )); then
    "$python_bin" "$repo_root/scripts/teleop/previous_sessions.py" --repo-root "$repo_root" --dry-run || die "Cannot inspect old project sessions"
    printf 'role=%s source=%s mode=%s replay=%s policy=M0-1010 record=%s hand_control=%s\n' \
        "$role" "$source_kind" "$run_mode" "$replay" "$record" "$hand_control"
    if (( dashboard )); then
        printf 'panes: publisher / retarget worker / deploy / dashboard (Three.js web visualization)\n'
    else
        printf 'panes: publisher / retarget worker / robot viewer :8081 / human viewer :8080 / deploy\n'
    fi
    printf 'Publisher: %s\n' "$(pane_command "$publisher_cmd")"
    printf 'Retarget worker: %s\n' "$(pane_command "$worker_cmd")"
    (( dashboard )) || printf 'Robot viewer: %s\n' "$(pane_command "$viewer_cmd")"
    if [[ "$source_kind" == mocap && "$hand_control" == on ]]; then
        printf 'Hand retarget worker: %s\n' "$(pane_command "$hand_worker_cmd")"
    fi
    (( dashboard )) || printf 'Human viewer: %s\n' "$(pane_command "$human_viewer_cmd")"
    printf 'Deployment policy: %s\n' "$(pane_command "$deploy_cmd")"
    (( ! pico_record )) || printf 'Recorder: %s\n' "$(pane_command "$record_cmd")"
    if (( dashboard )); then
        printf 'Dashboard: %s\n' "$(pane_command "$dashboard_cmd")"
        printf '  URL: %shttp://127.0.0.1:%s/%s  (listening on %s, Three.js single-canvas visualization)\n' \
            "$c_url" "$dashboard_port" "$c_reset" "$dashboard_bind"
        mapfile -t dash_addrs < <(dashboard_addrs)
        if ((${#dash_addrs[@]})); then
            printf '  Direct access: %shttp://%s:%s/%s\n' "$c_url" "${dash_addrs[0]}" "$dashboard_port" "$c_reset"
        fi
        if ((${#dash_addrs[@]})); then
            printf '  %sRemote access: bash scripts/dashboard_forward.sh start --host %s@%s%s\n' \
                "$c_dim" "${USER:-booster}" "${dash_addrs[0]}" "$c_reset"
        else
            printf '  %sRemote access: bash scripts/dashboard_forward.sh start --host <user>@<host>%s\n' \
                "$c_dim" "$c_reset"
        fi
        printf '  keyboard overlay: %s\n' "$([[ $dashboard_overlay == 1 ]] && echo on || echo 'off (pico backend ignores web r/y/b)')"
    fi
    if (( dashboard )); then
        printf 'Access: open http://<robot-ip>:%s/ directly; use bash scripts/dashboard_forward.sh start for remote access\n' "$dashboard_port"
    elif [[ "$run_mode" == sim ]]; then
        printf 'Viser SSH forwarding: -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 -L 8082:127.0.0.1:8082\n'
    else
        printf 'Viser SSH forwarding: -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081\n'
    fi
    print_reuse
    exit 0
fi

# One local launcher lock spans cleanup through pane creation (including --no-attach).
# Wrappers exec this entry, so no double cleanup. Lock is not inherited by panes.
command -v flock >/dev/null 2>&1 || die "flock is required"
launch_dir="${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}}/booster-teleop"
(umask 077; mkdir -p -- "$launch_dir") || die "Cannot create private launcher directory"
[[ -d "$launch_dir" && -O "$launch_dir" && ! -L "$launch_dir" ]] || die "Unsafe launcher directory"
exec {launch_lock}>>"$launch_dir/launch.lock"
flock -n "$launch_lock" || die "Another teleop launcher is running"
trap 'flock -u "$launch_lock"' EXIT
"$python_bin" "$repo_root/scripts/teleop/previous_sessions.py" --repo-root "$repo_root" || die "Old project session cleanup failed; new run not started"

PYTHONPATH="$repo_root" "$python_bin" -m booster_deploy.dashboard.shutdown_state "$diag_dir" "$session_name" start-run || die "Cannot clear previous business shutdown state"
BOOSTER_RUN_ORIGIN_NS="$BOOSTER_RUN_ORIGIN_NS" BOOSTER_TELEMETRY_EXECUTION="$run_mode" \
BOOSTER_TELEMETRY_TASK="$task" BOOSTER_TELEMETRY_INPUT_SOURCE="$source_kind" \
BOOSTER_TELEMETRY_MJCF="$telemetry_mjcf" BOOSTER_TELEMETRY_INPUT_FILE="$input_file" \
BOOSTER_TELEMETRY_TRAJECTORY_PATH="$trajectory_path" BOOSTER_TELEMETRY_TRAJECTORY_STEM="$trajectory_stem" \
BOOSTER_TELEMETRY_INPUT_PATH="$input_path" BOOSTER_TELEMETRY_REPLAY="$replay" \
BOOSTER_TELEMETRY_SN="$sn" BOOSTER_TELEMETRY_STARTED_AT="${session_timestamp:-$(date +%Y%m%d-%H%M%S)}" \
PYTHONPATH="$repo_root" "$python_bin" -c \
    'import sys; from booster_deploy.telemetry.runtime import initialize; initialize(sys.argv[1],directory=sys.argv[2],config=sys.argv[3],topology=sys.argv[4])' \
    "$session_name" "$diag_dir" "$telemetry_config" "$telemetry_topology" || die "Cannot initialize telemetry run"
command -v tmux >/dev/null 2>&1 || die "tmux is missing; prepare this machine with bash scripts/install_env.sh before launching teleoperation"
tmux has-session -t "$session_name" 2>/dev/null && die "tmux session already exists: $session_name"

# tmux needs a terminfo entry for $TERM on *this* machine. Over SSH from Ghostty the
# client sends TERM=xterm-ghostty, which a stock robot image does not know, and tmux
# aborts with "missing or unsuitable terminal: xterm-ghostty". Install the entry once
# (from the client: infocmp -x xterm-ghostty | ssh <robot> tic -x -) or let this
# fallback pick a TERM the machine already knows.
ensure_terminal() {
    local candidate=${TERM:-}
    if command -v infocmp >/dev/null 2>&1; then
        if [[ -n "$candidate" && "$candidate" != dumb ]] && infocmp "$candidate" >/dev/null 2>&1; then
            return 0
        fi
    elif [[ -n "$candidate" && "$candidate" != dumb ]]; then
        return 0
    fi
    echo "Note: TERM=${candidate:-unset} has no terminfo on this machine; using xterm-256color for tmux." >&2
    export TERM=xterm-256color
    if command -v infocmp >/dev/null 2>&1; then
        infocmp "$TERM" >/dev/null 2>&1 || die "xterm-256color terminfo is missing too; install ncurses-base"
    fi
}
ensure_terminal

# ------------------------------------------------------------------ tmux panes
session_pane=$(tmux new-session -d -s "$session_name" -c "$repo_root" ${session_size_args[@]+"${session_size_args[@]}"} -P -F '#{pane_id}' 'bash -i')
tmux set-option -t "$session_pane" @booster_diag_dir "$diag_dir"
tmux set-option -t "$session_pane" @booster_repo_root "$repo_root"
tmux set-option -t "$session_pane" @booster_python_env "$BOOSTER_PYTHON_ENV"
new_pane() {
    local pane target
    # Always split the LARGEST pane.  Splitting the first pane every time halves it
    # repeatedly (60 -> 30 -> 15 -> 7 -> 3 -> 1 rows) and the 6th pane fails with
    # "no space for new pane", which silently dropped the worker/viewer panes.
    target=$(tmux list-panes -t "$session_pane" -F '#{pane_id} #{pane_width} #{pane_height}' \
        | awk '{print $1, $2 * $3}' | sort -k2 -rn | head -1 | cut -d' ' -f1)
    if ! pane=$(tmux split-window -t "${target:-$session_pane}" -c "$repo_root" -P -F '#{pane_id}' 'bash -i' 2>&1); then
        die "cannot create a tmux pane (${pane:-no space}); the window is too small for this many panes. Start with --dashboard (240x60) or use a larger terminal."
    fi
    [[ "$pane" =~ ^%[0-9]+$ ]] || die "unexpected tmux pane id: $pane"
    printf '%s\n' "$pane"
}

worker_pane=$(new_pane)
robot_viewer_pane=""
hand_worker_pane=""
if [[ "$source_kind" == mocap && "$hand_control" == on ]]; then
    hand_worker_pane=$(new_pane)
fi
human_viewer_pane=""
if (( dashboard == 0 )); then
    robot_viewer_pane=$(new_pane)
    human_viewer_pane=$(new_pane)
fi
deploy_pane=$(new_pane)
record_pane=""
(( ! pico_record )) || record_pane=$(new_pane)
dashboard_pane=""
if (( dashboard )); then
    dashboard_pane=$(new_pane)
fi
tmux select-layout -t "$session_name" tiled >/dev/null
tmux set-option -t "$session_name" mouse on
tmux set-window-option -t "${session_name}:0" pane-border-status top >/dev/null
tmux set-window-option -t "${session_name}:0" pane-border-format ' #{pane_title} ' >/dev/null

tmux select-pane -t "$session_pane" -T "publisher ($replay $source_kind)"
tmux select-pane -t "$worker_pane" -T "retarget worker"
tmux set-option -p -t "$worker_pane" remain-on-exit on
tmux set-option -t "$session_name" @booster_worker_pane "$worker_pane"
if [[ -n "$robot_viewer_pane" ]]; then
    tmux select-pane -t "$robot_viewer_pane" -T "robot viewer :8081"
fi
if [[ -n "$hand_worker_pane" ]]; then
    tmux select-pane -t "$hand_worker_pane" -T "hand retarget worker"
fi
if [[ -n "$human_viewer_pane" ]]; then
    tmux select-pane -t "$human_viewer_pane" -T "human viewer :8080"
fi
tmux select-pane -t "$deploy_pane" -T "$deploy_title"
tmux set-option -t "$session_name" @booster_deploy_pane "$deploy_pane"
tmux set-option -t "$session_name" @booster_deploy_cmd "$(pane_command "$deploy_cmd")"
# Persist the exact existing offline commands; the dashboard never creates a source.
tmux set-option -t "$session_name" @booster_recording "$recording_path"
tmux set-option -t "$session_name" @booster_live_record "$record"
tmux set-option -t "$session_name" @booster_publisher_cmd "$(pane_command "$publisher_cmd")"
tmux set-option -t "$session_name" @booster_worker_cmd "$(pane_command "$worker_cmd")"
tmux set-option -t "$session_name" @booster_session_mode "$run_mode"
tmux set-option -t "$session_name" @booster_rt_reset_port "$rt_reset_port"
tmux set-option -t "$session_name" @booster_session_meta "$role/$source_kind/$run_mode/$replay"
if (( dashboard )); then
    # The dashboard reads this to tell the operator whether the web buttons can
    # drive the state machine in this session.
    tmux set-option -t "$session_name" @booster_dashboard_overlay "$dashboard_overlay"
fi
if [[ -n "$record_pane" ]]; then
    tmux select-pane -t "$record_pane" -T "raw PICO recorder"
fi
if [[ -n "$dashboard_pane" ]]; then
    tmux select-pane -t "$dashboard_pane" -T "dashboard :$dashboard_port"
fi


start_pane() {
    tmux send-keys -t "$1" -l "$(pane_command "$2")"
    tmux send-keys -t "$1" Enter
}
start_pane "$session_pane" "$publisher_cmd"
start_pane "$worker_pane" "$worker_cmd"
if [[ -n "$hand_worker_pane" ]]; then
    start_pane "$hand_worker_pane" "$hand_worker_cmd"
fi
start_pane "$deploy_pane" "$deploy_cmd"
[[ -z "$record_pane" ]] || start_pane "$record_pane" "$record_cmd"
if [[ -n "$dashboard_pane" ]]; then
    # Keep the pane (and its last output) after the server exits, so a crash is
    # visible in the session instead of silently disappearing.
    tmux set-option -p -t "$dashboard_pane" remain-on-exit on
    start_pane "$dashboard_pane" "$dashboard_cmd"
fi

if (( dashboard == 0 )); then
    start_pane "$robot_viewer_pane" "$viewer_cmd"
    start_pane "$human_viewer_pane" "$human_viewer_cmd"
fi
tmux select-window -t "${session_name}:0" >/dev/null 2>&1 || true
tmux select-pane -t "$deploy_pane"

# ------------------------------------------------------------------ summary
echo "Created tmux session: $session_name"
echo "  role=$role source=$source_kind mode=$run_mode replay=$replay policy=M0-1010 hand_control=$hand_control record=$record"
if (( dashboard )); then
    echo "  panes: publisher / retarget worker / deploy / dashboard"
    echo "  visualization: Three.js single canvas (human / retarget / sim), no viser panes"
else
    echo "  panes: publisher / retarget worker / robot viewer(:8081) / human viewer(:8080) / deploy"
fi
[[ "$record" != on ]] || echo "  trajectory recordings: $record_dir (see Data Recording and Replay in the README)"

if (( diag )); then
    echo "  diagnostics: 1 Hz [diag] lines per pane -> $diag_dir (<pane>.jsonl)"
    echo "               summary after the run: python scripts/diagnostics/summarize_session.py $diag_dir"
else
    echo "  diagnostics: off (--no-diag)"
fi
if (( dashboard )); then
    echo "  dashboard: ${c_url}http://127.0.0.1:$dashboard_port/${c_reset}   (local browser)"
    mapfile -t dash_addrs < <(dashboard_addrs)
    if ((${#dash_addrs[@]})); then
        echo "  Direct access (same network, no tunnel): ${c_url}http://${dash_addrs[0]}:$dashboard_port/${c_reset}"
        for addr in "${dash_addrs[@]:1:2}"; do
            echo "        ${c_dim}http://$addr:$dashboard_port/${c_reset}"
        done
    fi
    if ((${#dash_addrs[@]})); then
        echo "  ${c_dim}Remote access: bash scripts/dashboard_forward.sh start --host ${USER:-booster}@${dash_addrs[0]}${c_reset}"
    else
        echo "  ${c_dim}Remote access: bash scripts/dashboard_forward.sh start --host <user>@<host>${c_reset}"
    fi
    echo "             listening on $dashboard_bind:$dashboard_port — only this port is needed"
    if (( dashboard_overlay )); then
        echo "             keyboard overlay: ON (the pico backend also accepts web A/Y/B)"
    else
        echo "             keyboard overlay: off (web A/Y/B is disabled with pico; enable with --dashboard-keyboard-overlay)"
    fi
    dashboard_dead=$(tmux display-message -p -t "$dashboard_pane" "#{pane_dead}" 2>/dev/null || echo 0)
    if [[ "$dashboard_dead" == "1" ]]; then
        echo "  ${c_warn}dashboard process exited${c_reset} — inspect the dashboard pane (a port conflict is a common cause)"
    elif wait_dashboard "$dashboard_port"; then
        echo "  ${c_ok}ready${c_reset} healthz ok"
    else
        echo "  ${c_warn}starting${c_reset} healthz is not ready (no response within about 5s; refresh shortly)"
    fi
fi
if [[ "$replay" == offline ]]; then
    if [[ "$run_mode" == real ]]; then
        echo "  replay: press robot controller A for one cycle; clip end returns to Walking; B aborts"
    else
        echo "  replay: looping the clip (simulation)"
    fi
fi
if [[ "$run_mode" == sim ]] && (( dashboard == 0 )); then
    echo "  http://localhost:8082/ (MuJoCo), http://localhost:8080/ (human), http://localhost:8081/ (retargeted robot)"
fi
echo "Ctrl+C stops the current program and keeps the pane; Ctrl+b d detaches tmux."

print_reuse

flock -u "$launch_lock"
exec {launch_lock}>&-
trap - EXIT
(( attach )) || exit 0
if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$session_name"
else
    exec tmux attach-session -t "$session_name"
fi
