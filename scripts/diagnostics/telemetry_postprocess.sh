#!/usr/bin/env bash
# Single telemetry entry: build derived simulation offline, optionally serve the replay UI.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python=${BOOSTER_PYTHON_ENV:-$HOME/.venvs/booster_teleoperation}/bin/python
cli="$root/scripts/diagnostics/telemetry.py"
inventory="$root/scripts/diagnostics/telemetry_inventory.py"
report_python=${TELEMETRY_REPORT_PYTHON:-$python}
mode=""; target=""; live=0; dry=0; force=""; model=""; port=${TELEMETRY_PORT:-8123}
consistency=0; report_no_plots=0

usage() {
  cat <<'TXT'
唯一入口: telemetry_postprocess.sh

运行模式（二选一）:
  --slog <run目录>       处理一个 run
  --mlog <logs目录>     处理目录下的多个 run

默认动作:
  --slog 新真机离线且启用 telemetry 的 run：复现闭环仿真并生成完整报告。
                         unlimited run 生成多轨迹索引及各轨迹的一致性报告。
  --slog 旧 run：生成/复用 <run>/derived_simulation.npz。
  --mlog：体检并逐 run 生成/复用派生仿真。
  --live                 处理完成后启动只读回放 UI；默认 8123，占用时自动顺延
  --consistency          显式强制新 run 的完整复现模式（缺快照即拒绝）

参数:
  --no-plots             一致性报告不生成图；默认含 RMSE 热图
  --force                普通后处理时重算派生仿真
  --model PATH           仅普通离线仿真后处理使用；文件必须与内置 M0 ONNX 完全相同
  --port N               仅与 --live 相关
  --dry-run              只显示计划，不生成产物或启动 UI
  -h, --help             显示帮助

模式限制:
  --consistency 只能配合 --slog；SN/model/task/轨迹从 execution_config 读取，缺失则拒绝复现。
  --no-plots 只用于 --consistency；--force/--model 不能和 --consistency 同用。
  --dry-run 与 --live 不能同时使用。
  报告默认使用项目 venv；仅 TELEMETRY_REPORT_PYTHON 可显式覆盖，不自动切换到系统 Python。
TXT
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slog)  mode=single; target=${2:-}; shift 2 ;;
    --mlog)  mode=multi;  target=${2:-}; shift 2 ;;
    --live)  live=1; shift ;;
    --consistency) consistency=1; shift ;;
    --no-plots) report_no_plots=1; shift ;;
    --dry-run) dry=1; shift ;;
    --force) force="--force"; shift ;;
    --model) model=${2:-}; shift 2 ;;
    --port)  port=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$mode" && -n "$target" ]] || { usage; exit 2; }
command -v "$report_python" >/dev/null 2>&1 || { echo "report python is not executable: $report_python" >&2; exit 1; }
if [[ "$mode" == single && -s "$target/execution_config.json" ]]; then
  config_offline=$("$report_python" - "$target/execution_config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print('1' if c.get('offline') and c.get('telemetry_enabled') and c.get('execution') == 'real' else '0')
PY
  ) || { echo "cannot read execution_config.json" >&2; exit 4; }
  if [[ "$config_offline" == 1 ]]; then consistency=1; fi
fi
if [[ $dry -eq 1 && $live -eq 1 ]]; then
  echo "--dry-run 与 --live 互斥" >&2; exit 2
fi
if [[ $consistency -eq 1 ]]; then
  [[ "$mode" == single ]] || { echo "--consistency currently requires --slog" >&2; exit 2; }
  [[ -z "$force" && -z "$model" ]] || { echo "--force/--model cannot override the recorded run config in --consistency" >&2; exit 2; }
else
  [[ $report_no_plots -eq 0 ]] || { echo "--no-plots only applies to --consistency" >&2; exit 2; }
fi
[[ -x "$python" ]] || { echo "python venv missing: $python" >&2; exit 1; }
target=$(cd -- "$target" 2>/dev/null && pwd) || { echo "目录不存在: $target" >&2; exit 2; }
if [[ $consistency -eq 1 ]]; then
  [[ -f "$target/manifest.json" ]] || { echo "not a run directory: $target" >&2; exit 2; }
  exec_config="$target/execution_config.json"
  [[ -s "$exec_config" ]] || { echo "missing execution_config.json; refusing to guess run parameters" >&2; exit 4; }
  run_meta_text=$(PYTHONPATH="$root" "$report_python" -B - "$exec_config" <<'PY'
import json,sys
from pathlib import Path
from booster_deploy.telemetry.trajectory import REPORT_STEM
from booster_deploy.telemetry.run_config import provenance_path
config_path=Path(sys.argv[1]); c=json.loads(config_path.read_text()); run=config_path.parent
traj=c.get('trajectory') or {}; model=c.get('model') or {}; robot=c.get('robot_config') or {}
multi = c.get('offline') and c.get('trajectory_switch') == 'unlimited'
if multi:
    # Report naming must not depend on the last selection (which may never have played).
    traj = {'name': REPORT_STEM, 'stem': REPORT_STEM}
model_path=provenance_path(run, model.get('resolved_path'), model.get('repo_relative_path'))
mjcf_path=provenance_path(run, robot.get('mjcf') or c.get('mjcf'), robot.get('mjcf_repo_relative'))
for value in (c.get('sn') or 'unknown',traj.get('name'),traj.get('stem'),c.get('started_at'),model_path,c.get('task'),c.get('topology'),mjcf_path,'1' if c.get('telemetry_enabled') else '0',c.get('execution'),'1' if multi else '0'):
    print(value or '')
PY
  ) || { echo "cannot read recorded run metadata" >&2; exit 4; }
  mapfile -t run_meta <<< "$run_meta_text"
  run_sn=${run_meta[0]:-unknown}; trajectory_name=${run_meta[1]:-}; trajectory_stem=${run_meta[2]:-}
  started_at=${run_meta[3]:-}; resolved_model=${run_meta[4]:-}; report_task=${run_meta[5]:-}
  report_topology=${run_meta[6]:-}; report_mjcf=${run_meta[7]:-}; telemetry_on=${run_meta[8]:-0}; report_execution=${run_meta[9]:-}
  multi_trajectory=${run_meta[10]:-0}
  [[ "$run_sn" == unknown || "$run_sn" =~ ^[0-9]+$ ]] || { echo "execution_config SN must be digits or empty" >&2; exit 4; }
  [[ "$started_at" =~ ^[0-9]{8}-[0-9]{6}$ ]] || { echo "execution_config start time must be YYYYMMDD-HHMMSS" >&2; exit 4; }
  [[ -n "$trajectory_stem" && "$trajectory_stem" != . && "$trajectory_stem" != .. && "$trajectory_stem" != */* ]] || { echo "execution_config trajectory stem is unsafe" >&2; exit 4; }
  [[ -n "$trajectory_name" && -n "$trajectory_stem" && -n "$started_at" && -n "$resolved_model" && -n "$report_mjcf" ]] || { echo "execution_config lacks required trajectory/model/MJCF provenance" >&2; exit 4; }
  [[ "$report_task" == t2_gmc_rt && "$report_topology" == retarget ]] || { echo "consistency processing requires a T2 M0 retarget run" >&2; exit 4; }
  [[ $telemetry_on == 1 ]] || { echo "consistency processing requires telemetry enabled" >&2; exit 4; }
  [[ "$report_execution" == real ]] || { echo "consistency processing only applies to real execution logs" >&2; exit 4; }
  if [[ $dry -eq 0 ]]; then
    "$report_python" -B -c '
import importlib, shlex, sys
required_modules = ["numpy", "msgpack"] + (["matplotlib.pyplot"] if sys.argv[1] == "1" else [])
failed = []
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append((name, str(exc)))
if failed:
    print("报告依赖检查失败 / Report dependency check failed: " + sys.executable, file=sys.stderr)
    for name, error in failed:
        print("  " + name + ": " + error, file=sys.stderr)
    packages = " ".join(dict.fromkeys(name.split(".")[0] for name, _ in failed))
    print("确认后安装到该环境 / Install into this environment after approval:\n  "
          + shlex.quote(sys.executable) + " -m pip install " + packages, file=sys.stderr)
    sys.exit(4)
' "$((1 - report_no_plots))" || exit $?
  fi
  report_dir="$target/T2-${run_sn}-${trajectory_stem}-${started_at}_Report"
fi

build_one() {
  local run=$1
  local execution
  execution=$("$report_python" - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
config=json.loads((run/'execution_config.json').read_text()) if (run/'execution_config.json').is_file() else {}
manifest=json.loads((run/'manifest.json').read_text())
print(config.get('execution') or manifest.get('metadata',{}).get('execution',''))
PY
  ) || return 1
  if [[ "$execution" == sim ]]; then
    echo "  analyze $(basename "$run") (simulation already contains closed-loop state)"
    if [[ $dry -eq 0 ]]; then
      "$python" -B "$cli" analyze "$run" || return $?
    fi
    return 0
  fi
  if [[ $dry -eq 1 ]]; then
    echo "  plan   generate $(basename "$run")"
    return 0
  fi
  echo "  build  $(basename "$run")"
  local args=(simulate "$run")
  if [[ -n "$force" ]]; then args+=("$force"); fi
  if [[ -n "$model" ]]; then args+=(--model "$model"); fi
  "$python" -B "$cli" "${args[@]}" >/tmp/telemetry-postprocess.$$.log 2>&1 || {
    echo "  FAILED $(basename "$run") (see /tmp/telemetry-postprocess.$$.log)" >&2
    tail -3 /tmp/telemetry-postprocess.$$.log >&2
    return 1
  }
  echo "  ok     $(basename "$run")/derived_simulation.npz"
}

pick_online() {
  "$python" -B - "$1" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = json.loads((root / 'inventory.json').read_text())
usable = [r for r in rows if 'error' not in r and r.get('human', 0) > 0]
online = [r for r in usable if 'online' in r['run']] or usable
print(str(root / online[-1]['run']) if online else '')
PY
}

if [[ "$mode" == single ]]; then
  [[ -f "$target/manifest.json" ]] || { echo "不是 run 目录（缺 manifest.json）: $target" >&2; exit 2; }
  echo "== single: $target"
  if [[ $consistency -eq 1 ]]; then
    chosen=$target
    if [[ $dry -eq 1 ]]; then
      echo "  plan   simulate task=$report_task model=$resolved_model topology=$report_topology"
      if [[ $multi_trajectory == 1 ]]; then
        echo "  plan   report by committed trajectory -> $report_dir/index.html"
      else
        echo "  plan   report complete executions -> $report_dir"
      fi
    else
      [[ -f "$resolved_model" ]] || { echo "recorded model does not exist: $resolved_model" >&2; exit 4; }
      [[ -f "$report_mjcf" ]] || { echo "recorded MJCF does not exist: $report_mjcf" >&2; exit 4; }
      simulation_args=(simulate "$target" --output "$target/derived_simulation.npz" --force)
      echo "== recreate closed-loop simulation: task=$report_task model=$resolved_model"
      PYTHONPATH="$root" "$python" -B "$cli" "${simulation_args[@]}"
      PYTHONPATH="$root" "$python" -B "$cli" analyze "$target" --markdown >/dev/null
      report_args=(--run "$target" --sim-run "$target" --out "$report_dir" --robot-id "$run_sn" --trajectory-name "$trajectory_stem" --report-timestamp "$started_at" --mjcf "$report_mjcf")
      if [[ $report_no_plots -eq 1 ]]; then report_args+=(--no-plots); fi
      case "$report_dir" in "$target"/T2-*_Report) ;; *) echo "unsafe report directory: $report_dir" >&2; exit 4 ;; esac
      rm -rf -- "$report_dir"
      report_status=0
      "$report_python" -B "$root/scripts/diagnostics/robot_consistency_report.py" "${report_args[@]}" || report_status=$?
      if [[ $report_status != 0 ]]; then
        [[ $multi_trajectory == 1 && $report_status == 4 && -f "$report_dir/trajectory_index.json" ]] || exit "$report_status"
      fi
      cp "$target/execution_config.json" "$report_dir/execution_config.json"
      cp "$target/analysis.json" "$report_dir/telemetry_analysis.json"
      cp "$target/analysis.zh-CN.md" "$report_dir/telemetry_analysis.zh-CN.md"
      cp "$target/analysis.en.md" "$report_dir/telemetry_analysis.en.md"
      cp "$target/manifest.json" "$report_dir/manifest.json"
      if [[ $report_status != 0 ]]; then exit "$report_status"; fi
      chosen="$target"
    fi
  else
    build_one "$target"
    chosen=$target
  fi
else
  [[ -d "$target" ]] || { echo "不是目录: $target" >&2; exit 2; }
  echo "== inventory: $target"
  "$python" -B "$inventory" "$target" | tee "$target/inventory.txt"
  echo
  echo "== postprocess"
  failures=0
  while IFS= read -r run; do
    [[ -n "$run" ]] || continue
    build_one "$run" || failures=$((failures + 1))
  done < <("$python" -B - "$target" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for row in json.loads((root / 'inventory.json').read_text()):
    if 'error' in row or row.get('human', 0) <= 0:
        continue
    print(root / row['run'])
PY
)
  chosen=$(pick_online "$target")
  echo
  [[ $failures -eq 0 ]] || echo "warning: $failures run(s) failed; see logs" >&2
fi

if [[ $live -eq 1 ]]; then
  [[ $dry -eq 0 ]] || { echo "--dry-run 与 --live 互斥" >&2; exit 2; }
  [[ -n "${chosen:-}" && -f "$chosen/manifest.json" ]] || { echo "没有可打开的 run" >&2; exit 3; }
  echo
  echo "== live: $chosen"
  echo "ui    : http://127.0.0.1:${port}/"
  echo "tunnel: ssh -L ${port}:127.0.0.1:${port} <this-host>"
  echo "stop  : Ctrl-C"
  live_args=(serve "$chosen" --port "$port")
  if [[ -f "$chosen/derived_simulation.npz" ]]; then
    live_args+=(--generated-sim "$chosen/derived_simulation.npz")
  fi
  exec "$python" -B "$cli" "${live_args[@]}"
fi

echo
echo "done. 打开界面: bash scripts/diagnostics/telemetry_postprocess.sh --slog <run> --live"
