#!/usr/bin/env bash
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"

configure_python_index() {
    local timezone_offset
    timezone_offset=$(date +%z)
    if [[ "$timezone_offset" == "+0800" ]]; then
        export UV_INDEX_URL="${BOOSTER_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
        echo "UTC+8 detected; using PyPI index: $UV_INDEX_URL"
    fi
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
role=""
requested_env_path=""
assume_yes=0
check_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --role|--env-path)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            if [[ "$1" == --role ]]; then role=$2; else requested_env_path=$2; fi
            shift 2 ;;
        --yes) assume_yes=1; shift ;;
        --check-only) check_only=1; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/install_env.sh [--role host|onboard] [--env-path PATH] [--yes] [--check-only]
Without --role, choose the machine interactively.
host: teleoperation and simulation on Linux x86_64.
onboard: teleoperation and simulation on Linux AArch64.
--yes accepts installation/reuse prompts.
The onboard installer reuses a synchronized environment and repairs it only
when locked dependencies or the PICO SDK check fail.
--check-only verifies the onboard M0 teleoperation environment without installing
packages or starting robot control.
EOF
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$role" in ""|host|onboard) ;; *) echo "Invalid role: $role" >&2; exit 2 ;; esac

prompt() {
    local message=$1 answer
    (( assume_yes == 0 )) || return 0
    read -r -p "$message [y/N] " answer
    [[ "$answer" == y || "$answer" == Y || "$answer" == yes || "$answer" == YES ]]
}

load_onboard_ros_environment() {
    local ros_setup=/opt/booster/BoosterRos2Interface/install/setup.bash
    local source_result=0
    [[ -f "$ros_setup" ]] || {
        echo "Missing onboard ROS interface workspace: $ros_setup" >&2
        echo "Install/build BoosterRos2Interface before verifying the onboard environment." >&2
        return 1
    }
    echo "Loading onboard ROS interface workspace: $ros_setup"
    # Colcon-generated setup scripts may reference unset optional variables.
    set +u
    source "$ros_setup" || source_result=$?
    set -u
    return "$source_result"
}

install_tmux() {
    command -v apt-get >/dev/null 2>&1 || {
        echo "apt-get is required to install tmux; install it manually." >&2
        return 1
    }
    echo "tmux is missing. An internet connection and sudo access may be required."
    prompt "Install tmux using the system package manager?" || return 1
    sudo apt-get -o APT::Update::Error-Mode=any -o Acquire::Retries=0 \
        -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 update
    sudo apt-get install -y tmux
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux installation did not provide a tmux executable." >&2
        return 1
    }
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo "$(uv --version) is already installed: $(command -v uv)"
        return
    fi
    echo "uv is missing. An internet connection may be required."
    command -v curl >/dev/null 2>&1 || {
        echo "curl is required to download uv." >&2
        return 1
    }
    prompt "Install the latest stable uv into ~/.local/bin?" || return 1
    curl --fail --location --silent --show-error --retry 5 --max-time 600 \
        "https://astral.sh/uv/install.sh" | UV_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || {
        echo "uv installation did not provide an executable in PATH or ~/.local/bin." >&2
        return 1
    }
    uv --version
}

check_onboard_pico_sdk() {
    "$env_path/bin/python" - "$repo_root" <<'PY'
from pathlib import Path
from importlib.metadata import version
import hashlib
import sys
import tomllib
import xrobotoolkit_sdk as xrt

sdk_root = Path(sys.argv[1]) / "third_party/xrobotoolkit_sdk_aarch64"
expected_version = tomllib.loads((sdk_root / "python/pyproject.toml").read_text())["project"]["version"]
if version("xrobotoolkit-sdk") != expected_version:
    raise SystemExit(f"PICO SDK version must be {expected_version}")
required = (
    "init", "close", "register_frame_callback", "clear_frame_callback",
    "is_body_data_available", "get_body_joints_pose", "get_A_button",
    "get_B_button", "get_X_button", "get_Y_button",
)
missing = [name for name in required if not callable(getattr(xrt, name, None))]
if missing:
    raise SystemExit(f"PICO SDK is missing required callback/button APIs: {missing}")
extension = Path(xrt.__file__)
with extension.open("rb") as stream:
    header = stream.read(64)
if len(header) < 20 or header[:5] != b"\x7fELF\x02" or int.from_bytes(header[18:20], "little") != 183:
    raise SystemExit(f"PICO SDK extension is not a 64-bit AArch64 ELF: {extension}")
installed_library = extension.with_name("libPXREARobotSDK.so")
bundled_library = sdk_root / "lib/libPXREARobotSDK.so"
if hashlib.sha256(installed_library.read_bytes()).digest() != hashlib.sha256(bundled_library.read_bytes()).digest():
    raise SystemExit("Installed PICO native library differs from the bundled robot SDK")
print(f"PASS PICO SDK callback/button APIs and AArch64 extension: {extension}")
PY
}

check_onboard_packages() {
    local locked_requirements
    # Compare installed versions with the lock instead of local-package mtimes:
    # copying unchanged SDK sources onto the robot must not trigger a rebuild.
    locked_requirements=$(uv export --quiet --frozen --offline --no-header --extra onboard --no-hashes \
        --python "$env_path/bin/python" --no-emit-project --no-emit-package xrobotoolkit-sdk) || return 1
    "$env_path/bin/python" - "$locked_requirements" <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys
from packaging.requirements import Requirement

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Onboard environment requires Python 3.12")
failures = []
checked = 0
for line in sys.argv[1].splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    checked += 1
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        failures.append(f"Missing {requirement}")
        continue
    if installed not in requirement.specifier:
        failures.append(f"{requirement.name}: installed {installed}, need {requirement.specifier}")
if not checked or failures:
    raise SystemExit("\n".join(failures) or "No locked onboard requirements were exported")
print(f"PASS {checked} installed packages match the onboard lock")
PY
}

check_host_pico_service() {
    local service_root=/opt/apps/roboticsservice service_ldd
    [[ -x "$service_root/RoboticsServiceProcess" ]] || return 1
    file "$service_root/RoboticsServiceProcess" | grep -q 'ELF.*x86-64' || return 1
    service_ldd=$(LD_LIBRARY_PATH="$service_root:$service_root/lib:$service_root/SDK/x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$service_root/RoboticsServiceProcess" 2>&1) || {
        echo "$service_ldd" >&2
        return 1
    }
    if grep -q 'not found' <<<"$service_ldd"; then
        echo "$service_ldd" >&2
        return 1
    fi
}

install_host_pico_service() {
    local system_release package_name package_sha package_url package_path cache_dir
    local glib_package tool missing_tools=()
    local root_command=(sudo)
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
        echo "Host Pico service installation requires Linux x86_64." >&2
        return 1
    }
    command -v apt-get >/dev/null 2>&1 && command -v dpkg-deb >/dev/null 2>&1 || {
        echo "apt-get and dpkg-deb are required to install the host Pico service." >&2
        return 1
    }
    if [[ "$EUID" == 0 ]]; then
        root_command=()
    fi
    for tool in file pgrep python3 curl; do
        command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
    done
    if (( ${#missing_tools[@]} > 0 )); then
        echo "Installing host service tools (sudo access may be required): ${missing_tools[*]}"
        "${root_command[@]}" apt-get install -y file procps python3 curl || return 1
    fi
    if check_host_pico_service; then
        echo "Host Pico service is already installed: /opt/apps/roboticsservice/RoboticsServiceProcess"
        return
    fi

    system_release=$(. /etc/os-release && printf '%s:%s' "$ID" "$VERSION_ID") || return 1
    case "$system_release" in
        ubuntu:22.04)
            package_name=XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
            package_sha=61961067eb4b41f81ed7cae35f4690dbb0ddfefb329a12b24e0b90ebc46ada91
            glib_package=libglib2.0-0
            ;;
        ubuntu:24.04)
            package_name=XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
            package_sha=bce661f0be0b8a246ceecb2e5f1675a81c26b834648dc7fdf23f8c0bfe2a5d19
            glib_package=libglib2.0-0t64
            ;;
        *)
            echo "Automatic host Pico service installation supports Ubuntu 22.04/24.04; found $system_release." >&2
            return 1
            ;;
    esac
    package_url="https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/$package_name"
    cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/booster_teleoperation"
    package_path="${BOOSTER_PICO_SERVICE_DEB:-$cache_dir/$package_name}"
    if [[ ! -f "$package_path" ]]; then
        if [[ -n "${BOOSTER_PICO_SERVICE_DEB:-}" ]]; then
            echo "Specified host Pico service package does not exist: $package_path" >&2
            return 1
        fi
        mkdir -p "$cache_dir" || return 1
        echo "Downloading locked host Pico service v1.0.0: $package_url"
        curl --fail --location --retry 5 --max-time 600 "$package_url" -o "$package_path.part" || return 1
        mv "$package_path.part" "$package_path" || return 1
    fi
    printf '%s  %s\n' "$package_sha" "$package_path" | sha256sum --check --status || {
        echo "Host Pico service package SHA256 mismatch: $package_path" >&2
        return 1
    }
    [[ "$(dpkg-deb --field "$package_path" Package)" == roboticsservice && \
       "$(dpkg-deb --field "$package_path" Architecture)" == amd64 && \
       "$(dpkg-deb --field "$package_path" Version)" == 1.0.0.0 ]] || {
        echo "Expected roboticsservice 1.0.0.0 amd64 package: $package_path" >&2
        return 1
    }
    package_path=$(realpath "$package_path") || return 1
    echo "Installing host Pico service and system dependencies (sudo access may be required)."
    # The vendor package does not declare its postinst utilities or runtime libraries.
    "${root_command[@]}" apt-get install -y xdg-utils desktop-file-utils libgtk-3-bin \
        "$glib_package" libgssapi-krb5-2 libstdc++6 libgcc-s1 libzstd1 "$package_path" || return 1
    check_host_pico_service || {
        echo "Host Pico service failed executable/shared-library checks after installation." >&2
        return 1
    }
    echo "Host Pico service installed. Start it with: source scripts/env_set.sh"
}

install_host_pico_sdk() {
    local sdk_root="$repo_root/third_party/xrobotoolkit_sdk_x86_64"
    local sdk_library="$sdk_root/lib/libPXREARobotSDK.so"
    local sdk_sha=2348e1b0bf7f05cd13d95d628f04237b6b9fc50c6008b82f9bf4b046fc9373e6
    local root_command=(sudo)
    [[ -f "$sdk_library" ]] && file "$sdk_library" | grep -q 'ELF.*x86-64' || {
        echo "Host Pico SDK requires the bundled x86_64 library: $sdk_library" >&2
        return 1
    }
    printf '%s  %s\n' "$sdk_sha" "$sdk_library" | sha256sum --check --status || {
        echo "Bundled host Pico SDK SHA256 mismatch: $sdk_library" >&2
        return 1
    }
    if ! command -v g++ >/dev/null 2>&1; then
        [[ "$EUID" != 0 ]] || root_command=()
        echo "Installing host Pico SDK build tools (sudo access may be required)."
        "${root_command[@]}" apt-get install -y build-essential
    fi
    echo "Building/installing locked host Pico Python SDK 1.0.2 into $env_path"
    # Reuse the callback-enabled binding source with the bundled x86_64 library.
    # Build isolation uses the versions pinned in the binding's pyproject.toml.
    "$env_path/bin/python" - "$repo_root" "$sdk_root" <<'PY'
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]) / "third_party/xrobotoolkit_sdk_aarch64"
native = Path(sys.argv[2])
with tempfile.TemporaryDirectory(prefix="booster-pico-host-sdk-") as temporary:
    build = Path(temporary)
    shutil.copytree(source / "python", build / "python", ignore=shutil.ignore_patterns(
        "build", "dist", "*.egg-info", "__pycache__", "*.pyc",
    ))
    shutil.copytree(source / "include", build / "include")
    (build / "lib").mkdir()
    shutil.copy2(native / "lib/libPXREARobotSDK.so", build / "lib/libPXREARobotSDK.so")
    # Preserve both the binding's MIT license and the native library's Apache license.
    with (build / "python/LICENSE").open("a") as license_file:
        license_file.write("\n\nBundled native SDK library license:\n")
        license_file.write((native / "LICENSE").read_text())
    subprocess.run([
        "uv", "pip", "install", "--python", sys.executable,
        "--reinstall-package", "xrobotoolkit-sdk", "--no-cache", str(build / "python"),
    ], check=True)
PY
}

if [[ -z "$role" ]]; then
    echo "Choose deployment machine"
    echo "  1) Host computer (Linux x86_64)"
    echo "  2) Onboard computer (Linux aarch64)"
    read -r -p "Enter 1 or 2: " machine_choice
    case "$machine_choice" in
        1) role=host ;;
        2) role=onboard ;;
        *) echo "Enter 1 or 2." >&2; exit 2 ;;
    esac
fi
if (( check_only )) && [[ "$role" != onboard ]]; then
    echo "--check-only requires --role onboard." >&2
    exit 2
fi
[[ "$(uname -s)" == Linux ]] || { echo "Linux is required." >&2; exit 1; }
configure_python_index
case "$role" in
    host)
        [[ "$(uname -m)" == x86_64 ]] || { echo "Host setup requires x86_64." >&2; exit 1; }
        python_bin=3.12
        sync_extra=()
        ;;
    onboard)
        [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]] || { echo "Onboard setup requires aarch64." >&2; exit 1; }
        source "$repo_root/scripts/check_booster_version.sh"
        require_booster_version || exit 1
        python_bin=3.12
        sync_extra=(--extra onboard)
        ;;
esac
if (( check_only )); then
    command -v uv >/dev/null || { echo "uv is required for offline environment checks." >&2; exit 1; }
else
    install_uv
fi
if command -v tmux >/dev/null 2>&1; then
    echo "tmux is already installed: $(tmux -V)"
else
    (( check_only == 0 )) || { echo "tmux is missing from the onboard environment." >&2; exit 1; }
    install_tmux
fi

default_env_path="$HOME/.venvs/booster_teleoperation"
echo
if [[ -n "$requested_env_path" ]]; then
    env_path="$requested_env_path"
elif [[ "$role" == host && "$assume_yes" == 0 ]]; then
    echo "Choose virtual environment path"
    read -r -p "Enter path, or press Enter for $default_env_path: " env_path
    env_path=${env_path:-$default_env_path}
else
    echo "Use the default teleoperation virtual environment path"
    env_path="$default_env_path"
    echo "Environment path: $env_path"
fi
env_path=${env_path/#\~/$HOME}
if [[ -e "$env_path" ]] && ! prompt "Path exists: $env_path. Check and reuse this environment, synchronizing if needed?"; then
    echo "Environment setup cancelled." >&2
    exit 1
fi
echo "Selected role: $role"
echo "Selected environment: $env_path"

cd "$repo_root"
[[ -f "$repo_root/uv.lock" ]] || {
    echo "Missing uv.lock. Restore it from the repository or regenerate it in a networked maintainer environment." >&2
    exit 1
}
if [[ "$role" == host ]]; then
    install_host_pico_service
fi
if [[ ! -f "$env_path/pyvenv.cfg" ]]; then
    (( check_only == 0 )) || { echo "Onboard environment does not exist: $env_path" >&2; exit 1; }
    uv venv --python "$python_bin" "$env_path"
fi
if [[ "$role" == onboard ]]; then
    load_onboard_ros_environment
    export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
fi
reuse_onboard_environment=0
repair_onboard_sdk=0
if [[ "$role" == onboard ]]; then
    if ! check_onboard_pico_sdk; then
        repair_onboard_sdk=1
        echo "Installed PICO SDK is missing or incompatible with the bundled callback/button interface."
    fi
    if check_onboard_packages &&
        uv pip check --python "$env_path/bin/python" && (( repair_onboard_sdk == 0 )); then
        reuse_onboard_environment=1
        echo "Reusing synchronized onboard environment; dependencies and PICO button APIs are ready."
    else
        if (( check_only )); then
            echo "Onboard checks failed. Run install_env.sh without --check-only to repair the environment." >&2
            exit 1
        fi
        echo "Onboard environment is missing or out of sync; synchronizing locked dependencies."
    fi
fi
if [[ "$role" != onboard ]] || (( reuse_onboard_environment == 0 )); then
    sync_args=(--frozen --python "$env_path/bin/python" "${sync_extra[@]}")
    if [[ "$role" == onboard ]]; then
        if (( repair_onboard_sdk )); then
            # A damaged cached native wheel must not be reinstalled as the repair.
            uv cache clean xrobotoolkit-sdk
            sync_args+=(--reinstall-package xrobotoolkit-sdk)
        else
            # Preserve the verified SDK when only other packages need updating.
            sync_args+=(--inexact --no-install-package xrobotoolkit-sdk)
        fi
    fi
    UV_PROJECT_ENVIRONMENT="$env_path" uv sync "${sync_args[@]}"
fi
# uv sync removes packages outside the host lock; install the native binding after it.
if [[ "$role" == host ]]; then
    install_host_pico_sdk
fi
uv pip check --python "$env_path/bin/python"

if [[ "$role" == onboard ]]; then
    check_onboard_pico_sdk
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
"$env_path/bin/python" scripts/diagnostics/verify_environment.py "$role"
echo "Environment ready: source $env_path/bin/activate"
