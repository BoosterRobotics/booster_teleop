"""Analyze, regenerate and visualize one telemetry run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import webbrowser


COMMANDS = frozenset(("analyze", "simulate", "serve", "records"))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Write and print the telemetry integrity/lineage report")
    analyze.add_argument("directory", type=Path)
    analyze.add_argument("--markdown", action="store_true")
    analyze.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    records = sub.add_parser("records", help="Stream raw MessagePack records as readable JSON")
    records.add_argument("directory", type=Path)

    simulate = sub.add_parser("simulate", help="Regenerate a real run through its policy and MuJoCo")
    simulate.add_argument("directory", type=Path)
    simulate.add_argument("--output", type=Path)
    simulate.add_argument("--model", type=Path, help="Use a byte-identical copy of the bundled M0 ONNX for offline simulation")
    simulate.add_argument("--force", action="store_true")

    serve = sub.add_parser("serve", help="Start the synchronized read-only replay UI")
    serve.add_argument("directory", type=Path)
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8096)
    serve.add_argument("--execution", choices=("real", "sim"))
    serve.add_argument("--mjcf", type=Path)
    serve.add_argument("--generated-sim", type=Path)
    serve.add_argument("--no-generate-sim", action="store_true")
    serve.add_argument("--force-generate-sim", action="store_true")
    serve.add_argument("--open", action="store_true", help="Open the local URL in the default browser")
    return parser


def _raw_records(directory: Path):
    from .reader import iter_file
    def readable(value):
        if hasattr(value, "tolist"):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        if isinstance(value, bytes):
            return {"bytes": len(value), "hex": value.hex()}
        raise TypeError(type(value).__name__)
    errors = []
    for path in sorted(directory.glob("*.msgpack")):
        for row in iter_file(path, errors):
            print(json.dumps(row, ensure_ascii=False, default=readable))
    for error in errors:
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the former ``python -m ...cli DIRECTORY --markdown/--records`` entry.
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        if "--records" in argv:
            argv.remove("--records")
            argv.insert(0, "records")
        else:
            argv.insert(0, "analyze")
    args = _parser().parse_args(argv)
    directory = args.directory.expanduser().resolve()

    if args.command == "records":
        _raw_records(directory)
        return 0
    if args.command == "analyze":
        from .analysis import markdown, save
        report = save(directory)
        print(markdown(report, language="en") if args.markdown else
              json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "simulate":
        from .simulation import generate
        output = generate(directory, output=args.output, model=args.model, force=args.force)
        print(output)
        return 0

    generated = args.generated_sim.expanduser().resolve() if args.generated_sim else None
    if generated is None and not args.no_generate_sim:
        from .replay import ReplayRun
        # Legacy runs predate manifest metadata; ask the loader instead of guessing.
        execution = args.execution or ReplayRun(directory).execution
        if execution == "real":
            from .simulation import generate
            generated = generate(directory, force=args.force_generate_sim)
            print(f"Derived MuJoCo replay: {generated}", flush=True)

    from .server import serve
    server = None
    for candidate in [args.port] + [args.port + step for step in range(1, 10)]:
        try:
            server = serve(directory, bind=args.bind, port=candidate, execution=args.execution,
                           mjcf=args.mjcf, generated=generated)
            break
        except OSError as exc:
            if getattr(exc, 'errno', None) != 98:
                raise
            print(f"port {candidate} busy, trying {candidate + 1}", flush=True)
    if server is None:
        raise SystemExit(f'no free port near {args.port}')
    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{port}/"
    print(f"Telemetry replay: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
