from fnmatch import fnmatchcase
from pathlib import Path

from job_hunt.paths import WORKSPACE_ROOT


def get_workspace_root() -> Path:
    """Return the startup working directory as the sandbox boundary.

    The value is captured once when job_hunt starts, so later process cwd changes
    do not change the workspace for file tools or checkpoint selection.
    """
    return WORKSPACE_ROOT

# Paths matching any of these patterns are always denied
SENSITIVE_PATTERNS = [
    '*.env*',
    '*.pem',
    '*.key',
    '*.pfx',
    '*.p12',
    '*id_rsa*',
    '*id_ed25519*',
    '*id_ecdsa*',
    '*credentials*',
    '*secret*',
    '*password*',
    '*token*',
    '*.htpasswd',
    '.git/config',
    '*/.git/config',
    '.git-credentials',
    '/etc/passwd',
    '/etc/shadow',
    '~/.ssh/*',
    '~/.aws/*',
    '~/.gcloud/*',
]


def _normalize_for_match(value: str | Path) -> str:
    """Normalize path separators and case for cross-platform matching."""
    return str(value).replace('\\', '/').casefold()


def _find_sensitive_pattern(path: Path, workspace: Path) -> str | None:
    """Return the matching sensitive pattern, if any."""
    relative_path = path.relative_to(workspace)
    candidates = (
        _normalize_for_match(relative_path),
        _normalize_for_match(path),
    )

    for pattern in SENSITIVE_PATTERNS:
        pattern_path = Path(pattern).expanduser() if pattern.startswith('~') else pattern
        normalized_pattern = _normalize_for_match(pattern_path)
        if any(fnmatchcase(candidate, normalized_pattern) for candidate in candidates):
            return pattern

    return None


def validate_path(file_path: str | Path) -> tuple[Path, str]:
    """Validate that *file_path* is safe to read.

    Returns (resolved_path, error).
    On success, error is an empty string.
    """
    raw = str(file_path)
    path = Path(raw).expanduser().resolve()

    # Deny paths outside the workspace (catches path traversal: ../../etc/passwd)
    workspace = get_workspace_root()
    try:
        path.relative_to(workspace)
    except ValueError:
        return path, f'path is outside workspace ({workspace}): {path}'

    # Deny sensitive files. Match both relative and absolute paths so patterns
    # such as .git/config work consistently on Windows and Unix-like systems.
    sensitive_pattern = _find_sensitive_pattern(path, workspace)
    if sensitive_pattern:
        return path, (
            f'sensitive file blocked by pattern "{sensitive_pattern}": {path}'
        )

    return path, ''
