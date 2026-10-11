"""Small helpers for resolving one video or a directory of videos."""

from pathlib import Path


VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mkv", ".mov", ".m4v"})


def resolve_source_path(source: str) -> Path:
    """Resolve a directory argument, including the project's legacy data path."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path

    # The supplied dataset may be placed in ``videos`` while older checkouts
    # keep the same files in ``data/video``.
    if path == Path("videos"):
        for legacy_path in (Path("data/video"), Path(__file__).resolve().parent / "data/video"):
            if legacy_path.is_dir():
                return legacy_path
    return path


def find_video_files(directory: Path) -> list[Path]:
    """Return supported video files in deterministic order."""
    if not directory.exists():
        raise FileNotFoundError(f"Video directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a video directory: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )
