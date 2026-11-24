#!/usr/bin/env bash
# One-command SSH tunnel for the teleop dashboard.
#
# Why this exists: `ssh -L 8095:127.0.0.1:8095 ...` typed by hand fails in three
# ways we hit repeatedly:
#   * a leftover forward keeps the local port busy, so a host-mode dashboard can
#     no longer bind ("Address already in use");
#   * the remote side has nothing listening, so every browser retry prints
#     "channel N: open failed: connect failed: Connection refused";
#   * the port differs per session and gets mistyped.
#
# Prefer a direct connection when the robot is routable (the dashboard already
# binds 0.0.0.0) — this script is the fallback for cross-subnet / remote access.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/dashboard_forward.sh [start|status|stop] [options]

  start          Create a tunnel and run a health check (default action)
  status         Show tunnel and dashboard status
  stop           Close the tunnel (no leftover forwarding)

  --host USER@HOST     Onboard host; default: $BOOSTER_ROBOT_HOST or booster@172.16.68.150
  --port N             Onboard dashboard port; default: 8095
  --local-port N       Local port; defaults to --port; automatically changes if busy
  --open               Open a browser on success
  --wait SEC           Maximum health-check wait in seconds; default: 8
EOF
}

action="start"
# The board's address changes between sessions (172.16.42.48 today, 172.16.68.150
# before), so do not hardcode one: BOOSTER_ROBOT_HOST wins, otherwise probe.
host="${BOOSTER_ROBOT_HOST:-}"
host_candidates=(booster@172.16.42.48 booster@172.16.68.150)
remote_port=8095
local_port=""
open_browser=0
wait_s=8

while (($#)); do
    case "$1" in
        start|status|stop) action=$1 ;;
        --host) host=$2; shift ;;
        --port) remote_port=$2; shift ;;
        --local-port) local_port=$2; shift ;;
        --open) open_browser=1 ;;
        --wait) wait_s=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 64 ;;
    esac
    shift
done

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    c_reset=$'\033[0m'; c_ok=$'\033[1;32m'; c_err=$'\033[1;31m'
    c_dim=$'\033[2m'; c_url=$'\033[1;36m'; c_warn=$'\033[1;33m'
else
    c_reset=''; c_ok=''; c_err=''; c_dim=''; c_url=''; c_warn=''
fi

say()  { printf '%s\n' "$*"; }
die()  { printf '%serror:%s %s\n' "$c_err" "$c_reset" "$*" >&2; exit 1; }
warn() { printf '%swarn:%s %s\n' "$c_warn" "$c_reset" "$*" >&2; }

ssh_base=(ssh -o BatchMode=yes -o ConnectTimeout=6 -o LogLevel=ERROR)

resolve_host() {
    if [[ -n "$host" ]]; then
        return 0
    fi
    local candidate
    for candidate in "${host_candidates[@]}"; do
        if ssh -o BatchMode=yes -o ConnectTimeout=4 -o LogLevel=ERROR "$candidate" true \
                >/dev/null 2>&1; then
            host=$candidate
            return 0
        fi
    done
    die "Cannot find the onboard host: tried ${host_candidates[*]}; specify --host user@ip or export BOOSTER_ROBOT_HOST"
}

remote_listening() {
    "${ssh_base[@]}" "$host" "ss -ltn 2>/dev/null | grep -qE '(:|\.)${remote_port}[[:space:]]'" >/dev/null 2>&1
}

local_port_busy() {
    ss -ltn 2>/dev/null | grep -qE "(:|\.)${1}[[:space:]]"
}

pick_local_port() {
    local candidate=$1
    local limit=$((candidate + 20))
    while ((candidate < limit)); do
        if ! local_port_busy "$candidate"; then
            printf '%s' "$candidate"
            return 0
        fi
        candidate=$((candidate + 1))
    done
    return 1
}

socket_for() { printf '%s/booster-dashboard-%s.sock' "${TMPDIR:-/tmp}" "$1"; }
tunnel_alive() { ssh -S "$1" -O check "$host" >/dev/null 2>&1; }

health_url() { printf 'http://127.0.0.1:%s/healthz' "$1"; }

wait_health() {
    local port=$1 deadline=$((SECONDS + wait_s))
    while ((SECONDS < deadline)); do
        if curl -fsS --max-time 1 "$(health_url "$port")" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

open_if_needed() {
    local url=$1
    ((open_browser)) || return 0
    if command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1 || true
    else warn "Neither open nor xdg-open was found; open $url manually"
    fi
}

main_start() {
    resolve_host
    # 1) Refuse to tunnel into nothing: that is what produced the endless
    #    "connect failed: Connection refused" lines.
    if ! "${ssh_base[@]}" "$host" true >/dev/null 2>&1; then
        die "Cannot SSH to $host (key authentication or network issue); first verify that ssh $host can log in"
    fi
    if ! remote_listening; then
        printf '%sNo process on onboard host %s is listening on :%s%s -- dashboard is not running.\n' \
            "$c_err" "$host" "$remote_port" "$c_reset" >&2
        cat >&2 <<EOF

Start a session on the onboard host first (or start the dashboard separately):

  ssh $host
  cd ~/Workspace/booster_whole_body_teleoperation
  bash scripts/teleop_tmux.sh onboard pico sim --replay offline --dashboard

  If the session already exists and only the dashboard needs starting:
  python scripts/dashboard.py --session <session-name> --bind 0.0.0.0 --port $remote_port

EOF
        exit 2
    fi

    # 2) Reuse an existing tunnel of ours, otherwise find a free local port.
    local port="${local_port:-$remote_port}"
    local sock
    sock=$(socket_for "$port")
    if [[ -S "$sock" ]] && tunnel_alive "$sock"; then
        if wait_health "$port"; then
            say "${c_ok}Existing tunnel${c_reset} 127.0.0.1:$port → $host:$remote_port"
            say "  URL: ${c_url}http://127.0.0.1:$port/${c_reset}"
            open_if_needed "http://127.0.0.1:$port/"
            return 0
        fi
        warn "Tunnel exists but the dashboard is unresponsive; recreating it"
        ssh -S "$sock" -O exit "$host" >/dev/null 2>&1 || true
        rm -f "$sock"
    fi

    if local_port_busy "$port"; then
        if ss -ltnp 2>/dev/null | grep -qE "(:|\.)${port}[[:space:]].*sshd?:"; then
            warn "Local port $port is busy with another SSH forward (likely a leftover); using the next free port"
        else
            warn "Local port $port is busy with another process; using the next free port"
        fi
        port=$(pick_local_port $((port + 1))) || die "Cannot find a free local port"
    fi
    sock=$(socket_for "$port")

    # 3) ControlMaster + -N so start/status/stop are deterministic and no forward
    #    survives the shell that created it.
    "${ssh_base[@]}" -M -S "$sock" -fN \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
        -L "${port}:127.0.0.1:${remote_port}" "$host" \
        || die "Failed to create tunnel (ssh -L $port:127.0.0.1:$remote_port $host)"

    if ! wait_health "$port"; then
        ssh -S "$sock" -O exit "$host" >/dev/null 2>&1 || true
        rm -f "$sock"
        die "Tunnel created but $(health_url "$port") is unresponsive (dashboard may still be starting; retry start later or increase --wait)"
    fi

    say "${c_ok}Tunnel ready${c_reset} 127.0.0.1:$port → $host:$remote_port"
    say "  URL: ${c_url}http://127.0.0.1:$port/${c_reset}"
    say "  ${c_dim}Stop: scripts/dashboard_forward.sh stop --local-port $port${c_reset}"
    open_if_needed "http://127.0.0.1:$port/"
}

main_status() {
    resolve_host
    local port="${local_port:-$remote_port}"
    local sock
    sock=$(socket_for "$port")
    if [[ -S "$sock" ]] && tunnel_alive "$sock"; then
        say "Tunnel: ${c_ok}running${c_reset} (127.0.0.1:$port → $host:$remote_port)"
    else
        say "Tunnel: not running (local port $port)"
    fi
    if remote_listening; then
        say "Remote listener: ${c_ok}yes${c_reset} :$remote_port"
    else
        say "Remote listener: ${c_err}no${c_reset} :$remote_port (dashboard is not running)"
    fi
    if wait_health "$port"; then
        say "Health check: ${c_ok}ok${c_reset} $(health_url "$port")"
    else
        say "Health check: ${c_dim}failed${c_reset} $(health_url "$port")"
    fi
}

main_stop() {
    resolve_host
    local port="${local_port:-$remote_port}"
    local sock
    sock=$(socket_for "$port")
    if [[ -S "$sock" ]]; then
        ssh -S "$sock" -O exit "$host" >/dev/null 2>&1 || true
        rm -f "$sock"
        say "Tunnel closed (local port $port)"
    else
        say "No tunnel created by this script (local port $port)"
    fi
}

case "$action" in
    start) main_start ;;
    status) main_status ;;
    stop) main_stop ;;
    *) usage; exit 64 ;;
esac
