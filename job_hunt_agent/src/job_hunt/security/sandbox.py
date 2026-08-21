import os
from fnmatch import fnmatch
from pathlib import Path

def get_workspace_root() -> Path:
    """Return the current working directory as the sandbox boundary.

    Evaluated at call time so changing directory between agent runs
    updates the sandbox range. This is the same approach Claude Code uses —
    the workspace is wherever you launched it.
    """
    return Path.cwd().resolve()

# Paths matching any of these patterns are always denied
SENSITIVE_PATTERNS = [
    '*.env',
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
    '.git-credentials',
    '/etc/passwd',
    '/etc/shadow',
    '~/.ssh/*',
    '~/.aws/*',
    '~/.gcloud/*',
]


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

    # Deny sensitive files
    normalized = str(path)
    for pattern in SENSITIVE_PATTERNS:
        check = os.path.expanduser(pattern) if pattern.startswith('~') else pattern
        if fnmatch(normalized, check):
            return path, f'sensitive file blocked by pattern "{pattern}": {path}'

    return path, ''
