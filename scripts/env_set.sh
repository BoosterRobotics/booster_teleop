#!/usr/bin/env bash

# Unified environment entry point.
# Interactive: source scripts/env_set.sh
# Automation: source scripts/env_set.sh host|onboard pico|mocap|offline [local IP] [server IP]

if [[ -n "${BASH_VERSION:-}" ]]; then
    _ENV_SET_SOURCE_FILE="${BASH_SOURCE[0]}"
    [[ "${BASH_SOURCE[0]}" != "$0" ]] || { echo "Source this file: source scripts/env_set.sh" >&2; exit 1; }
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _ENV_SET_SOURCE_FILE="${(%):-%N}"
    [[ "${ZSH_EVAL_CONTEXT:-}" == *:file* ]] || { echo "Source this file: source scripts/env_set.sh" >&2; return 1; }
else
    echo "Use Bash or Zsh to source scripts/env_set.sh" >&2
    return 1 2>/dev/null || exit 1
fi

_env_set_fail() { echo "env_set: $*" >&2; return 1; }
_env_set_prompt() {
    local label="$1" default_value="$2" answer
    printf "%s [%s]: " "$label" "$default_value" >&2
    IFS= read -r answer || return 1
    printf "%s" "${answer:-$default_value}"
}
_env_set_ip_valid() {
    "$TELEOP_PYTHON" - "$1" "$2" <<'PY'
from ipaddress import IPv4Address
import sys
try:
    IPv4Address(sys.argv[2])
except ValueError:
    raise SystemExit(f"Invalid {sys.argv[1]} IPv4 address: {sys.argv[2]}")
PY
}
_env_set_prepend_path() {
    local entry="$1" current_path="$2"
    case ":$current_path:" in
        *":$entry:"*) printf '%s' "$current_path" ;;
        *) printf '%s' "$entry${current_path:+:$current_path}" ;;
    esac
}
_env_set_source_file() {
    local had_nounset=0 source_result
    case "$-" in *u*) had_nounset=1; set +u ;; esac
    if source "$1"; then source_result=0; else source_result=$?; fi
    (( had_nounset == 0 )) || set -u
    return "$source_result"
}
_env_set_wireless_ip() {
    "$TELEOP_PYTHON" - <<'PY'
import fcntl
from pathlib import Path
import socket
import struct
found = False
for interface in sorted(Path("/sys/class/net").iterdir()):
    if not ((interface / "wireless").exists() or (interface / "phy80211").exists()):
        continue
    try:
        if (interface / "operstate").read_text().strip() != "up":
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            request = struct.pack("256s", interface.name.encode()[:15])
            address = fcntl.ioctl(sock.fileno(), 0x8915, request)
        print(f"\033[32m  WIRELESS_IP={socket.inet_ntoa(address[20:24])} ({interface.name})\033[0m")
        found = True
        break
    except OSError:
        continue
if not found:
    print("  WIRELESS_IP=unavailable (no active Wi-Fi interface with IPv4)")
PY
}
_env_set_service_ready() {
    "$TELEOP_PYTHON" - "$1" "$2" <<'PY'
import socket, sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=.25):
    pass
PY
} >/dev/null 2>&1
_env_set_check_elf() {
    local binary="$1" architecture="$2" description="$3" require_exec="${4:-0}" output
    if [[ "$require_exec" == 1 ]]; then
        [[ -x "$binary" ]] || { _env_set_fail "$description is missing or not executable: $binary"; return 1; }
    else
        [[ -f "$binary" ]] || { _env_set_fail "$description is missing: $binary"; return 1; }
    fi
    output=$(file "$binary" 2>&1) || { echo "$output" >&2; return 1; }
    [[ "$output" == *"$architecture"* ]] || { _env_set_fail "$description has the wrong architecture: $output"; return 1; }
    output=$(LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" ldd "$binary" 2>&1) || { echo "$output" >&2; return 1; }
    if [[ "$output" == *"not found"* ]]; then
        echo "$description has missing shared libraries:" >&2
        echo "$output" >&2
        return 1
    fi
}
_env_set_start_service() {
    local service_root="$1" service_bin="$1/RoboticsServiceProcess"
    local service_host="$2" service_port="$3" service_log="$4" library_path="$5" attempt
    local service_arch=x86-64
    [[ "$_ENV_SET_ROLE" != onboard ]] || service_arch='ARM aarch64'
    if _env_set_service_ready "$service_host" "$service_port"; then
        echo "Reusing RoboticsServiceProcess at $service_host:$service_port"
        return 0
    fi
    if ! _env_set_service_ready "$service_host" "$service_port"; then
        if pgrep -f '[R]oboticsServiceProcess' >/dev/null 2>&1; then
            echo "RoboticsServiceProcess is running; waiting for $service_host:$service_port"
        else
            _env_set_check_elf "$service_bin" "$service_arch" RoboticsServiceProcess 1 || return 1
            echo "Starting RoboticsServiceProcess: $service_bin"
            (cd "$service_root" && LD_LIBRARY_PATH="$library_path" QT_QPA_PLATFORM=offscreen \
                nohup "$service_bin" >"$service_log" 2>&1 </dev/null &) || return 1
        fi
        for attempt in {1..40}; do
            _env_set_service_ready "$service_host" "$service_port" && break
            sleep .25
        done
    fi
    _env_set_service_ready "$service_host" "$service_port" || {
        echo "RoboticsServiceProcess is not ready: $service_host:$service_port" >&2
        [[ ! -f "$service_log" ]] || tail -n 80 "$service_log" >&2
        return 1
    }
}
_env_set_cleanup() {
    unset _ENV_SET_SOURCE_FILE _ENV_SET_SCRIPT_DIR _ENV_SET_REPO_ROOT
    unset _ENV_SET_ROLE _ENV_SET_MODE _ENV_SET_LOCAL_IP _ENV_SET_SERVER_IP
    unset _ENV_SET_ENV_PATH _ENV_SET_UPSTREAM _ENV_SET_SERVICE_ROOT
    unset _ENV_SET_DEFAULT_ROLE _ENV_SET_ROS_SETUP
    unset _ENV_SET_NATIVE_ARCH _ENV_SET_SERVICE_SDK_ARCH _ENV_SET_DEFAULT_LOG
    unset _ENV_SET_SERVICE_AUTOSTART
    unset _ENV_SET_LIBRARY_DIR
    unset _ENV_SET_NO_PROXY _ENV_SET_NO_PROXY_ENTRY
    unset -f _env_set_fail _env_set_prompt _env_set_ip_valid _env_set_prepend_path
    unset -f _env_set_source_file _env_set_wireless_ip _env_set_service_ready
    unset -f _env_set_check_elf _env_set_start_service _env_set_cleanup
}

_ENV_SET_SCRIPT_DIR="$(cd "$(dirname "$_ENV_SET_SOURCE_FILE")" && pwd)" || return 1
_ENV_SET_REPO_ROOT="$(cd "$_ENV_SET_SCRIPT_DIR/.." && pwd)" || return 1
_ENV_SET_ROLE="${1:-}"
_ENV_SET_MODE="${2:-}"

if [[ -z "$_ENV_SET_ROLE" ]]; then
    echo "Step 1/2: choose deployment machine" >&2
    echo "  1) Host computer (Linux x86_64)" >&2
    echo "  2) Onboard computer (Linux aarch64)" >&2
    _ENV_SET_DEFAULT_ROLE=1
    [[ "$(uname -m)" != aarch64 && "$(uname -m)" != arm64 ]] || _ENV_SET_DEFAULT_ROLE=2
    _ENV_SET_ROLE=$(_env_set_prompt "Enter 1 for Host or 2 for Onboard" "$_ENV_SET_DEFAULT_ROLE") || { _env_set_cleanup; return 1; }
    case "$_ENV_SET_ROLE" in 1) _ENV_SET_ROLE=host ;; 2) _ENV_SET_ROLE=onboard ;; *) _env_set_fail "choose 1 or 2"; _env_set_cleanup; return 1 ;; esac
elif [[ "$_ENV_SET_ROLE" == 1 ]]; then _ENV_SET_ROLE=host
elif [[ "$_ENV_SET_ROLE" == 2 ]]; then _ENV_SET_ROLE=onboard
fi
case "$_ENV_SET_ROLE" in host|onboard) ;; *) _env_set_fail "role must be host or onboard"; _env_set_cleanup; return 1 ;; esac

if [[ -z "$_ENV_SET_MODE" ]]; then
    echo "Step 2/2: choose input or offline environment" >&2
    echo "  1) Mocap" >&2
    echo "  2) Pico" >&2
    echo "  3) Offline (no live input service)" >&2
    _ENV_SET_MODE=$(_env_set_prompt "Enter 1 for Mocap, 2 for Pico, or 3 for Offline" "1") || { _env_set_cleanup; return 1; }
    case "$_ENV_SET_MODE" in 1) _ENV_SET_MODE=mocap ;; 2) _ENV_SET_MODE=pico ;; 3) _ENV_SET_MODE=offline ;; *) _env_set_fail "choose 1, 2, or 3"; _env_set_cleanup; return 1 ;; esac
fi
case "$_ENV_SET_MODE" in 1) _ENV_SET_MODE=mocap ;; 2) _ENV_SET_MODE=pico ;; 3) _ENV_SET_MODE=offline ;; esac
case "$_ENV_SET_MODE" in mocap|pico|offline) ;; *) _env_set_fail "mode must be mocap, pico, or offline"; _env_set_cleanup; return 1 ;; esac

[[ "$(uname -s)" == Linux ]] || { _env_set_fail "Linux is required"; _env_set_cleanup; return 1; }
if [[ "$_ENV_SET_ROLE" == host ]]; then
    [[ "$(uname -m)" == x86_64 ]] || { _env_set_fail "Host requires x86_64"; _env_set_cleanup; return 1; }
else
    [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]] || { _env_set_fail "Onboard requires AArch64"; _env_set_cleanup; return 1; }
fi

_ENV_SET_ENV_PATH="${BOOSTER_PYTHON_ENV:-${VIRTUAL_ENV:-$HOME/.venvs/booster_teleoperation}}"
[[ -x "$_ENV_SET_ENV_PATH/bin/python" && -f "$_ENV_SET_ENV_PATH/bin/activate" ]] || {
    _env_set_fail "Python environment not found: $_ENV_SET_ENV_PATH; run bash scripts/install_env.sh"
    _env_set_cleanup; return 1
}
_env_set_source_file "$_ENV_SET_ENV_PATH/bin/activate" || { _env_set_cleanup; return 1; }
export BOOSTER_PYTHON_ENV="$(cd "$_ENV_SET_ENV_PATH" && pwd)"
export TELEOP_PYTHON="$BOOSTER_PYTHON_ENV/bin/python"
export BOOSTER_DEPLOY_ROOT="$_ENV_SET_REPO_ROOT"
PYTHONPATH="$(_env_set_prepend_path "$_ENV_SET_REPO_ROOT" "${PYTHONPATH:-}")"
export PYTHONPATH

if [[ "$_ENV_SET_MODE" == mocap ]]; then
    _ENV_SET_LOCAL_IP="${3:-${BOOSTER_MOCAP_LOCAL_IP:-192.168.10.102}}"
    _ENV_SET_SERVER_IP="${4:-${BOOSTER_MOCAP_SERVER_IP:-192.168.10.199}}"
    if [[ -z "${3:-}" ]]; then _ENV_SET_LOCAL_IP=$(_env_set_prompt "Mocap local IP" "$_ENV_SET_LOCAL_IP") || { _env_set_cleanup; return 1; }; fi
    if [[ -z "${4:-}" ]]; then _ENV_SET_SERVER_IP=$(_env_set_prompt "Mocap server IP" "$_ENV_SET_SERVER_IP") || { _env_set_cleanup; return 1; }; fi
    _env_set_ip_valid local "$_ENV_SET_LOCAL_IP" || { _env_set_cleanup; return 1; }
    _env_set_ip_valid server "$_ENV_SET_SERVER_IP" || { _env_set_cleanup; return 1; }
    export BOOSTER_MOCAP_LOCAL_IP="$_ENV_SET_LOCAL_IP" BOOSTER_MOCAP_SERVER_IP="$_ENV_SET_SERVER_IP"
fi

export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/config/dds_configs/single-board_configs/fastdds_profile_bridge.xml
export FASTDDS_DEFAULT_PROFILES_FILE=/opt/booster/config/dds_configs/single-board_configs/fastdds_profile_bridge.xml
export BOOSTER_RT_REFERENCE_PORT=5556 BOOSTER_RT_REFERENCE_TOPIC=retarget_rt.reference_50hz
export BOOSTER_RT_REFERENCE_TRANSITION_S="${BOOSTER_RT_REFERENCE_TRANSITION_S:-1.5}"
export BOOSTER_PICO_CONTROL_PORT=5555 BOOSTER_PICO_CONTROL_TOPIC=pico_rt.control
export BOOSTER_PICO_CONTROL_STALE_TIMEOUT="${BOOSTER_PICO_CONTROL_STALE_TIMEOUT:-0.5}"
export BOOSTER_PICO_BUTTON_DEBOUNCE="${BOOSTER_PICO_BUTTON_DEBOUNCE:-0.25}"
export BOOSTER_PICO_TRIGGER_DEADZONE="${BOOSTER_PICO_TRIGGER_DEADZONE:-0.01}"
export BOOSTER_MOCAP_LOCAL_PORT="${BOOSTER_MOCAP_LOCAL_PORT:-8002}" BOOSTER_MOCAP_SERVER_PORT="${BOOSTER_MOCAP_SERVER_PORT:-8080}"
export BOOSTER_MOCAP_HUMAN_FORMAT="${BOOSTER_MOCAP_HUMAN_FORMAT:-bvh}" BOOSTER_MOCAP_ROBOT_NAME="${BOOSTER_MOCAP_ROBOT_NAME:-t2}"
export BOOSTER_MOCAP_POLL_HZ="${BOOSTER_MOCAP_POLL_HZ:-500}"
export BOOSTER_MOCAP_PROVIDER_TIMEOUT_S="${BOOSTER_MOCAP_PROVIDER_TIMEOUT_S:-30}"
export BOOSTER_MOCAP_CALIBRATION_POLL_HZ="${BOOSTER_MOCAP_CALIBRATION_POLL_HZ:-100}"
export BOOSTER_MOCAP_CALIBRATION_TIMEOUT_S="${BOOSTER_MOCAP_CALIBRATION_TIMEOUT_S:-240}"
export BOOSTER_MOCAP_RT_BIND="${BOOSTER_MOCAP_RT_BIND:-0.0.0.0}" BOOSTER_MOCAP_RT_HOST="${BOOSTER_MOCAP_RT_HOST:-127.0.0.1}"
export BOOSTER_MOCAP_RT_PORT="${BOOSTER_MOCAP_RT_PORT:-5557}" BOOSTER_MOCAP_RT_TOPIC="${BOOSTER_MOCAP_RT_TOPIC:-mocap_rt.retarget_input}"
export BOOSTER_MOCAP_RT_SEND_HWM="${BOOSTER_MOCAP_RT_SEND_HWM:-1}" BOOSTER_MOCAP_RT_MAX_CACHED_FRAMES="${BOOSTER_MOCAP_RT_MAX_CACHED_FRAMES:-4}"
export BOOSTER_MOCAP_RT_PRINT_INTERVAL="${BOOSTER_MOCAP_RT_PRINT_INTERVAL:-300}"
export BOOSTER_RT_REFERENCE_HOST="${BOOSTER_RT_REFERENCE_HOST:-127.0.0.1}" BOOSTER_PICO_CONTROL_HOST="${BOOSTER_PICO_CONTROL_HOST:-127.0.0.1}"
export BOOSTER_RT_RESET_HOST="${BOOSTER_RT_RESET_HOST:-127.0.0.1}" BOOSTER_RT_RESET_BIND="${BOOSTER_RT_RESET_BIND:-0.0.0.0}"
export BOOSTER_RT_RESET_PORT="${BOOSTER_RT_RESET_PORT:-5559}" BOOSTER_RT_RESET_TOPIC="${BOOSTER_RT_RESET_TOPIC:-retarget_rt.reset}"
export BOOSTER_HAND_RETARGET_HOST="${BOOSTER_HAND_RETARGET_HOST:-127.0.0.1}"
export BOOSTER_HAND_RETARGET_PORT=5558 BOOSTER_HAND_RETARGET_TOPIC=hand_retarget.revo2_commands
# Consumers choose joint-name defaults after discovering each hand type.
# Leave BOOSTER_LEFT/RIGHT_HAND_JOINT_NAMES unset unless explicitly configured.
export BOOSTER_ROBOT_JOYSTICK_TOPIC="${BOOSTER_ROBOT_JOYSTICK_TOPIC:-/remote_controller_state}"
export BOOSTER_ROBOT_JOYSTICK_STALE_TIMEOUT="${BOOSTER_ROBOT_JOYSTICK_STALE_TIMEOUT:-0.5}"
export BOOSTER_HAND_PUBLISH_RATE_HZ=50 BOOSTER_HAND_TRIGGER_EPS=0.01
if [[ "$_ENV_SET_MODE" == pico ]]; then
    export BOOSTER_HAND_OPEN_POSITION="${BOOSTER_HAND_OPEN_POSITION:-0,1.3089969389957472,0,0,0,0}"
    export BOOSTER_HAND_CLOSED_POSITION="${BOOSTER_HAND_CLOSED_POSITION:-1.4835298641951802,1.3089969389957472,1.4835298641951802,1.7453292519943296,1.7453292519943296,1.7453292519943296}"
else
    export BOOSTER_HAND_OPEN_POSITION="${BOOSTER_HAND_OPEN_POSITION:-0,0,0,0,0,0}"
    export BOOSTER_HAND_CLOSED_POSITION="${BOOSTER_HAND_CLOSED_POSITION:-1.3089969389957472,0.10471975511965977,1.7453292519943296,1.7453292519943296,1.7453292519943296,1.7453292519943296}"
fi
export BOOSTER_HAND_CLOSE_THUMB_DELAY_S="${BOOSTER_HAND_CLOSE_THUMB_DELAY_S:-1.0}"
export BOOSTER_HAND_RETARGET_STALE_S="${BOOSTER_HAND_RETARGET_STALE_S:-0.25}"
export BOOSTER_LEFT_HAND_TOPIC="${BOOSTER_LEFT_HAND_TOPIC:-/hand_left/left_revo2_hand_position_controller/commands}"
export BOOSTER_RIGHT_HAND_TOPIC="${BOOSTER_RIGHT_HAND_TOPIC:-/hand_right/right_revo2_hand_position_controller/commands}"
export BOOSTER_HAND_TURBO_ENABLED="${BOOSTER_HAND_TURBO_ENABLED:-${BOOSTER_HAND_SOFT_TURBO:-1}}"
export BOOSTER_LEFT_HAND_FEEDBACK_TOPIC="${BOOSTER_LEFT_HAND_FEEDBACK_TOPIC:-/hand_left/joint_states}"
export BOOSTER_RIGHT_HAND_FEEDBACK_TOPIC="${BOOSTER_RIGHT_HAND_FEEDBACK_TOPIC:-/hand_right/joint_states}"
export BOOSTER_HAND_TURBO_FEEDBACK_STALE_S="${BOOSTER_HAND_TURBO_FEEDBACK_STALE_S:-0.15}"
export BOOSTER_HAND_TURBO_WINDOW_FRAMES="${BOOSTER_HAND_TURBO_WINDOW_FRAMES:-5}"
export BOOSTER_HAND_TURBO_STATIONARY_TOLERANCE="${BOOSTER_HAND_TURBO_STATIONARY_TOLERANCE:-0.0017453292519943296}"
export BOOSTER_HAND_TURBO_STATIONARY_DURATION_S="${BOOSTER_HAND_TURBO_STATIONARY_DURATION_S:-0.2}"
export BOOSTER_HAND_TURBO_CLOSE_ERROR_MIN="${BOOSTER_HAND_TURBO_CLOSE_ERROR_MIN:-0.017453292519943295}"
export BOOSTER_HAND_TURBO_BACKOFF_UNITS="${BOOSTER_HAND_TURBO_BACKOFF_UNITS:-0.0017453292519943296}"
export BOOSTER_HAND_TURBO_RETRY_PERIOD_S="${BOOSTER_HAND_TURBO_RETRY_PERIOD_S:-0.5}"
unset BOOSTER_HAND_VELOCITY BOOSTER_HAND_EFFORT
if [[ ( "${BOOSTER_ENV_MODE:-}" == pico && "${BOOSTER_REMOTE_CONTROL_BACKEND:-}" == pico ) || \
      ( ( "${BOOSTER_ENV_MODE:-}" == mocap || "${BOOSTER_ENV_MODE:-}" == offline ) && "${BOOSTER_REMOTE_CONTROL_BACKEND:-}" == keyboard ) ]]; then
    unset BOOSTER_REMOTE_CONTROL_BACKEND
fi
if [[ "$_ENV_SET_MODE" == pico ]]; then
    export BOOSTER_REMOTE_CONTROL_BACKEND="${BOOSTER_REMOTE_CONTROL_BACKEND:-pico}"
    export BOOSTER_HAND_TRIGGER_CONTROL=1 BOOSTER_HAND_RETARGET_CONTROL=0
    unset BOOSTER_HAND_MAPPING_CONFIG
elif [[ "$_ENV_SET_MODE" == mocap ]]; then
    export BOOSTER_REMOTE_CONTROL_BACKEND="${BOOSTER_REMOTE_CONTROL_BACKEND:-keyboard}"
    export BOOSTER_HAND_TRIGGER_CONTROL=0 BOOSTER_HAND_RETARGET_CONTROL=1
    export BOOSTER_HAND_MAPPING_CONFIG="${BOOSTER_HAND_MAPPING_CONFIG:-$_ENV_SET_REPO_ROOT/booster_deploy/hand_retargeting/resources/t2_revo2_ros_mapping.yaml}"
else
    export BOOSTER_REMOTE_CONTROL_BACKEND="${BOOSTER_REMOTE_CONTROL_BACKEND:-keyboard}"
    export BOOSTER_HAND_TRIGGER_CONTROL=0 BOOSTER_HAND_RETARGET_CONTROL=0
    unset BOOSTER_HAND_MAPPING_CONFIG
fi
case "$BOOSTER_REMOTE_CONTROL_BACKEND" in
    auto|keyboard|joystick|joystick-robot|pico) ;;
    *) _env_set_fail "unsupported BOOSTER_REMOTE_CONTROL_BACKEND: $BOOSTER_REMOTE_CONTROL_BACKEND"; _env_set_cleanup; return 1 ;;
esac

if [[ "$_ENV_SET_ROLE" == onboard ]]; then
    _ENV_SET_UPSTREAM="${BOOSTER_UPSTREAM_HOST:-127.0.0.1}"
    export BOOSTER_RT_REFERENCE_HOST="$_ENV_SET_UPSTREAM" BOOSTER_RT_RESET_HOST="$_ENV_SET_UPSTREAM" BOOSTER_HAND_RETARGET_HOST="$_ENV_SET_UPSTREAM"
    export BOOSTER_PICO_CONTROL_HOST="$_ENV_SET_UPSTREAM"
    _ENV_SET_ROS_SETUP=/opt/booster/BoosterRos2Interface/install/setup.bash
    if [[ -n "${ZSH_VERSION:-}" && -f /opt/booster/BoosterRos2Interface/install/setup.zsh ]]; then
        _ENV_SET_ROS_SETUP=/opt/booster/BoosterRos2Interface/install/setup.zsh
    fi
    [[ -f "$_ENV_SET_ROS_SETUP" ]] || { _env_set_fail "Missing $_ENV_SET_ROS_SETUP"; _env_set_cleanup; return 1; }
    _env_set_source_file "$_ENV_SET_ROS_SETUP" || { _env_set_cleanup; return 1; }
    # Use the onboard bridge profile for every ROS process.  Fast DDS renamed
    # this variable, so export both the current and legacy spellings.
    export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/config/dds_configs/single-board_configs/fastdds_profile_bridge.xml
    export FASTDDS_DEFAULT_PROFILES_FILE=/opt/booster/config/dds_configs/single-board_configs/fastdds_profile_bridge.xml
    _env_set_source_file "$BOOSTER_PYTHON_ENV/bin/activate" || { _env_set_cleanup; return 1; }
    export PYTHONPATH="$(_env_set_prepend_path "$_ENV_SET_REPO_ROOT" "${PYTHONPATH:-}")"
else
    export BOOSTER_RT_REFERENCE_HOST=127.0.0.1 BOOSTER_RT_RESET_HOST=127.0.0.1 BOOSTER_HAND_RETARGET_HOST=127.0.0.1
    export BOOSTER_PICO_CONTROL_HOST=127.0.0.1 BOOSTER_MOCAP_RT_HOST=127.0.0.1
fi

export BOOSTER_PICO_HOST_ENV_READY=0 BOOSTER_PICO_ONBOARD_ENV_READY=0
if [[ "$_ENV_SET_MODE" == pico ]]; then
    if [[ "$_ENV_SET_ROLE" == onboard ]]; then
        export ROBOTICS_SERVICE_ROOT="$_ENV_SET_REPO_ROOT/third_party/roboticsservice_aarch64"
        export XROBOT_SDK_ROOT="${XROBOT_SDK_ROOT:-$_ENV_SET_REPO_ROOT/third_party/xrobotoolkit_sdk_aarch64}"
        export XROBOT_SDK_LIB_DIR="${XROBOT_SDK_LIB_DIR:-$XROBOT_SDK_ROOT/lib}"
        export XROBOT_SERVICE_COMPAT_LIB_DIR="${XROBOT_SERVICE_COMPAT_LIB_DIR:-$_ENV_SET_REPO_ROOT/third_party/xrobotoolkit_service_compat_aarch64/lib}"
        _ENV_SET_NATIVE_ARCH='ARM aarch64'
        _ENV_SET_SERVICE_SDK_ARCH=arm64
    else
        export ROBOTICS_SERVICE_ROOT="${ROBOTICS_SERVICE_ROOT:-/opt/apps/roboticsservice}"
        export XROBOT_SDK_ROOT="${XROBOT_SDK_ROOT:-$_ENV_SET_REPO_ROOT/third_party/xrobotoolkit_sdk_x86_64}"
        export XROBOT_SDK_LIB_DIR="${XROBOT_SDK_LIB_DIR:-$XROBOT_SDK_ROOT/lib}"
        _ENV_SET_NATIVE_ARCH=x86-64
        _ENV_SET_SERVICE_SDK_ARCH=x64
    fi
    export PXREA_SDK_LIBRARY="${PXREA_SDK_LIBRARY:-$XROBOT_SDK_LIB_DIR/libPXREARobotSDK.so}"
    # Prepend in reverse order to keep the SDK ahead of vendor service libraries.
    for _ENV_SET_LIBRARY_DIR in "${XROBOT_SERVICE_COMPAT_LIB_DIR:-}" \
        "$ROBOTICS_SERVICE_ROOT/SDK/$_ENV_SET_SERVICE_SDK_ARCH" "$ROBOTICS_SERVICE_ROOT/lib" \
        "$ROBOTICS_SERVICE_ROOT" "$XROBOT_SDK_LIB_DIR"; do
        [[ -z "$_ENV_SET_LIBRARY_DIR" ]] || LD_LIBRARY_PATH="$(_env_set_prepend_path "$_ENV_SET_LIBRARY_DIR" "${LD_LIBRARY_PATH:-}")"
    done
    export LD_LIBRARY_PATH
    _ENV_SET_NO_PROXY="${no_proxy:-${NO_PROXY:-}}"
    for _ENV_SET_NO_PROXY_ENTRY in 127.0.0.1 localhost ::1; do
        case ",$_ENV_SET_NO_PROXY," in
            *",$_ENV_SET_NO_PROXY_ENTRY,"*) ;;
            *) _ENV_SET_NO_PROXY="${_ENV_SET_NO_PROXY:+$_ENV_SET_NO_PROXY,}$_ENV_SET_NO_PROXY_ENTRY" ;;
        esac
    done
    no_proxy="$_ENV_SET_NO_PROXY" NO_PROXY="$_ENV_SET_NO_PROXY"
    export no_proxy NO_PROXY
    if [[ -n "${http_proxy:-}${https_proxy:-}${HTTP_PROXY:-}${HTTPS_PROXY:-}" ]]; then
        echo "env_set: http(s)_proxy is set; loopback bypasses it via no_proxy=$no_proxy." >&2
    fi
    _env_set_check_elf "$PXREA_SDK_LIBRARY" "$_ENV_SET_NATIVE_ARCH" 'XRoboToolkit SDK' || { _env_set_cleanup; return 1; }
    "$TELEOP_PYTHON" -c 'import xrobotoolkit_sdk' >/dev/null 2>&1 || {
        _env_set_fail "Cannot import xrobotoolkit_sdk using $TELEOP_PYTHON; run bash scripts/install_env.sh"
        _env_set_cleanup; return 1
    }
    export QT_PLUGIN_PATH="$(_env_set_prepend_path "$ROBOTICS_SERVICE_ROOT/plugins" "${QT_PLUGIN_PATH:-}")"
    export QT_QML_PATH="$(_env_set_prepend_path "$ROBOTICS_SERVICE_ROOT/qml" "${QT_QML_PATH:-}")"
    export QT_QPA_PLATFORM=offscreen
    export BOOSTER_ROBOTICS_SERVICE_HOST="${BOOSTER_ROBOTICS_SERVICE_HOST:-127.0.0.1}"
    export BOOSTER_ROBOTICS_SERVICE_PORT="${BOOSTER_ROBOTICS_SERVICE_PORT:-60061}"
    _ENV_SET_DEFAULT_LOG=/tmp/booster-roboticsservice-start.log
    [[ "$_ENV_SET_ROLE" != host ]] || _ENV_SET_DEFAULT_LOG=/tmp/booster-roboticsservice-host-start.log
    export BOOSTER_ROBOTICS_SERVICE_LOG="${BOOSTER_ROBOTICS_SERVICE_LOG:-$_ENV_SET_DEFAULT_LOG}"
    if "$TELEOP_PYTHON" - "$BOOSTER_ROBOTICS_SERVICE_HOST" "$BOOSTER_ROBOTICS_SERVICE_PORT" <<'CHECK_ENDPOINT'
from ipaddress import IPv4Address
import sys
try:
    address = IPv4Address(sys.argv[1])
    if not address.is_loopback or not 1 <= int(sys.argv[2]) <= 65535:
        raise ValueError
except ValueError:
    raise SystemExit("Robotics service endpoint must use a loopback IPv4 address and port 1..65535")
CHECK_ENDPOINT
    then
        :
    else
        _env_set_cleanup
        return 1
    fi
    _ENV_SET_SERVICE_AUTOSTART="${BOOSTER_PICO_AUTOSTART_SERVICE:-1}"
    case "$_ENV_SET_SERVICE_AUTOSTART" in
        1|true|TRUE|yes|YES|on|ON)
            _ENV_SET_SERVICE_AUTOSTART=1
            _env_set_start_service "$ROBOTICS_SERVICE_ROOT" "$BOOSTER_ROBOTICS_SERVICE_HOST" "$BOOSTER_ROBOTICS_SERVICE_PORT" "$BOOSTER_ROBOTICS_SERVICE_LOG" "$LD_LIBRARY_PATH" || { _env_set_cleanup; return 1; }
            ;;
        0|false|FALSE|no|NO|off|OFF) _ENV_SET_SERVICE_AUTOSTART=0 ;;
        *) _env_set_fail "BOOSTER_PICO_AUTOSTART_SERVICE must be 1/0 or true/false"; _env_set_cleanup; return 1 ;;
    esac
    if [[ "$_ENV_SET_SERVICE_AUTOSTART" == 1 ]]; then
        if [[ "$_ENV_SET_ROLE" == host ]]; then export BOOSTER_PICO_HOST_ENV_READY=1; else export BOOSTER_PICO_ONBOARD_ENV_READY=1; fi
    fi
fi
export BOOSTER_ENV_ROLE="$_ENV_SET_ROLE" BOOSTER_ENV_MODE="$_ENV_SET_MODE"
echo "Loaded $_ENV_SET_ROLE $_ENV_SET_MODE environment:"
echo "  TELEOP_PYTHON=$TELEOP_PYTHON"
echo "  BOOSTER_RT_REFERENCE_HOST=$BOOSTER_RT_REFERENCE_HOST:$BOOSTER_RT_REFERENCE_PORT"
if [[ "$_ENV_SET_MODE" == mocap ]]; then
    echo "  MOCAP_LOCAL=$BOOSTER_MOCAP_LOCAL_IP:$BOOSTER_MOCAP_LOCAL_PORT/udp"
    echo "  MOCAP_SERVER=$BOOSTER_MOCAP_SERVER_IP:$BOOSTER_MOCAP_SERVER_PORT"
fi
if [[ "$_ENV_SET_MODE" == pico ]]; then
    echo "  ROBOTICS_SERVICE_ROOT=$ROBOTICS_SERVICE_ROOT"
    if [[ "$_ENV_SET_SERVICE_AUTOSTART" == 1 ]]; then
        echo "  ROBOTICS_SERVICE_PID=$(pgrep -f '[R]oboticsServiceProcess' | tr '\n' ' ')"
    fi
    echo "  LOCAL_SDK_ENDPOINT=$BOOSTER_ROBOTICS_SERVICE_HOST:$BOOSTER_ROBOTICS_SERVICE_PORT"
    _env_set_wireless_ip
    echo "Set the Pico application connection target to this machine's LAN IP."
fi
_env_set_cleanup
return 0
