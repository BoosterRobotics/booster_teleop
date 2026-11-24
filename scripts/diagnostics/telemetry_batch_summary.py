"""Summarize per-run --slog reports without sharing simulation references."""
import argparse
from html import escape
import json
from pathlib import Path
from statistics import median
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, help='directory containing telemetry runs')
    args = parser.parse_args()
    runs = []
    for path in args.logs.glob('*/execution_config.json'):
        if not (path.parent / 'manifest.json').is_file():
            continue
        config = json.loads(path.read_text())
        if config.get('offline') and config.get('execution') == 'real' and config.get('telemetry_enabled'):
            runs.append((path.parent, config))
    if not runs:
        parser.error('no real offline telemetry runs found')

    rows, rmse = [], []
    reported = 0
    for run, config in sorted(runs, key=lambda item: ((item[1].get('trajectory') or {}).get('stem') or '', item[0])):
        multi = config.get('trajectory_switch') == 'unlimited'
        stem = 'multi-trajectory' if multi else (config.get('trajectory') or {}).get('stem')
        report = run / f"T2-{config.get('sn') or 'unknown'}-{stem}-{config.get('started_at')}_Report"
        page = report / ('index.html' if multi else 'report-summary.html')
        name = escape(stem or run.name)
        run_label = escape(run.name)
        if not page.is_file():
            rows.append(f'<tr><td>{name}</td><td>{run_label}</td><td colspan="5">缺少报告，请先运行 --slog</td></tr>')
            continue
        reported += 1
        link = f'<a href="{quote(page.relative_to(args.logs).as_posix())}">{name}</a>'
        metrics = sorted(report.glob('*/metrics.json')) if not multi else []
        if not metrics:
            rows.append(f'<tr><td>{link}</td><td>{run_label}</td><td colspan="5">见运行级报告</td></tr>')
        for path in metrics:
            result = json.loads(path.read_text())
            data = result['summary']
            rmse.append(data['rmse_median_deg'])
            rows.append(f'<tr><td>{link}</td><td>{run_label}</td><td>{escape(result["robot"])}</td>'
                        f'<td>{data["window_lineage_coverage"]:.1%}</td><td>{data["frames"]}</td>'
                        f'<td>{data["rmse_median_deg"]:.2f}</td><td>{data["rmse_worst_deg"]:.2f}</td></tr>')

    output = args.logs / 'batch-report.html'
    aggregate = f'；{len(rmse)} 轮的关节 RMSE 中位数再取中位数：{median(rmse):.2f}°' if rmse else ''
    output.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真机离线批测总报告</title>'
        '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}'
        'td,th{border:1px solid #aaa;padding:.4rem;text-align:left}</style>'
        '<h1>真机离线批测总报告</h1>'
        f'<p>已生成单 run 报告：{reported}/{len(runs)}{aggregate}。</p>'
        '<p>各轮使用所属 run 的仿真，不跨轨迹配对。多轨迹 run 的明细见链接。</p>'
        '<table><thead><tr><th>轨迹</th><th>Run</th><th>轮次</th><th>来源覆盖率</th><th>配对帧</th>'
        '<th>关节 RMSE 中位数 (°)</th><th>最差关节 RMSE (°)</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table></html>\n', encoding='utf-8')
    print(f'{reported}/{len(runs)} run reports, {len(rmse)} execution metrics -> {output}')


if __name__ == '__main__':
    main()
