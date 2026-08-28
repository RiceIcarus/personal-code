from pathlib import Path

from langchain_core.tools import tool

from job_hunt.security.sandbox import get_workspace_root, validate_path

IGNORED_DIRECTORIES = {
    '.git',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '__pycache__',
    'build',
    'checkpoints',
    'dist',
    '.venv',
}


def _is_ignored_directory(path: Path) -> bool:
    """Return whether a directory contains generated or internal data."""
    return any(part.casefold() in IGNORED_DIRECTORIES for part in path.parts)


def _display_path(path: Path, workspace: Path) -> str:
    """Return a workspace-relative path for tool output."""
    relative = path.relative_to(workspace)
    return relative.as_posix() or '.'


@tool
def list_files(
    directory: str = '.',
    recursive: bool = False,
    max_depth: int = 2,
    max_results: int = 100,
) -> str:
    """List visible files and directories within the workspace.

    Args:
        directory: Directory to inspect, relative or absolute. Defaults to the workspace root.
        recursive: Whether to include entries below the requested directory.
        max_depth: Maximum recursion depth when recursive is enabled.
        max_results: Maximum number of entries to return.

    Returns:
        A workspace-relative listing. Sensitive entries and generated runtime directories
        are omitted.
    """
    if max_depth < 1 or max_depth > 10:
        return 'Error: max_depth must be between 1 and 10'
    if max_results < 1 or max_results > 1000:
        return 'Error: max_results must be between 1 and 1000'

    workspace = get_workspace_root()
    path, error = validate_path(directory)
    if error:
        return f'Error: {error}'
    if not path.exists():
        return f'Error: directory not found — {path}'
    if not path.is_dir():
        return f'Error: not a directory — {path}'
    if _is_ignored_directory(path):
        return f'Error: directory is not available for listing — {path}'

    entries: list[str] = []
    pending = [(path, 0)]

    while pending and len(entries) < max_results:
        current, depth = pending.pop(0)
        try:
            children = sorted(
                current.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.casefold()),
            )
        except OSError as exc:
            return f'Error: failed to list directory — {current}, reason: {exc}'

        for child in children:
            if child.is_dir() and child.name.casefold() in IGNORED_DIRECTORIES:
                continue

            _, child_error = validate_path(child)
            if child_error:
                continue

            child_path = _display_path(child, workspace)
            if child.is_symlink():
                entries.append(f'[LINK] {child_path}')
                continue

            if child.is_dir():
                entries.append(f'[DIR]  {child_path}/')
                if recursive and depth + 1 < max_depth:
                    pending.append((child, depth + 1))
                continue

            try:
                size = child.stat().st_size
            except OSError:
                size = None
            size_text = f' ({size:,} bytes)' if size is not None else ''
            entries.append(f'[FILE] {child_path}{size_text}')

            if len(entries) >= max_results:
                break

    result = [f'Visible entries under {_display_path(path, workspace)}:']
    result.extend(entries)
    if pending or len(entries) >= max_results:
        result.append(f'[Listing truncated at {max_results} entries]')
    if not entries:
        result.append('(no visible entries)')
    return '\n'.join(result)
