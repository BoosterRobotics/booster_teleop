"""Choose the newest compatible offline input, using existing publisher loaders.

Only automatic defaults use this search. Explicit --recording remains authoritative
and is handled by the launcher as before. No preview/model/retarget validation.
"""
from pathlib import Path


def latest_recording(repo_root: Path, source: str) -> Path:
    if source == "pico":
        from .pico.offline_recording import load_pico_raw_recording as load
        folder, suffix = repo_root / "logs/pico_raw", ".npz"
    elif source == "mocap":
        from .mocap.offline_publisher import load_recorded_mocap_bvh as load
        folder, suffix = repo_root / "logs/mocap_raw", ".bvh"
    else:
        raise ValueError(f"Unsupported input source: {source}")
    candidates = []
    if folder.is_dir():
        for path in folder.iterdir():
            try:
                if path.is_file() and path.suffix.lower() == suffix:
                    candidates.append((path.stat().st_mtime_ns, str(path), path))
            except OSError:
                continue
    for _, _, path in sorted(candidates, reverse=True):
        try:
            load(path)
        except Exception:
            # A newer incomplete/incompatible clip must not hide an older valid one.
            continue
        return path.resolve()
    raise ValueError(f"No compatible {source} {suffix} recording in {folder}; pass --recording PATH. / "
                     "No compatible recordings; pass --recording PATH (the trajectory path will not be prompted)")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source", choices=("pico", "mocap"), required=True)
    args = parser.parse_args()
    try:
        print(latest_recording(args.repo_root, args.source))
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
