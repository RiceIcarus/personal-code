from pathlib import Path
import re

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path.cwd().resolve()


def _find_project_root() -> Path:
    """Find the job_hunt source project root from the installed package path."""
    for parent in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        pyproject = parent / 'pyproject.toml'
        if pyproject.exists() and (
            'name = "job-hunt"' in pyproject.read_text(encoding='utf-8')
        ):
            return parent

    return PACKAGE_ROOT


PROJECT_ROOT = _find_project_root()
ENV_PATH = PROJECT_ROOT / '.env'
CHECKPOINT_ROOT = PROJECT_ROOT / 'checkpoints'
SYSTEM_PROMPT_PATH = PACKAGE_ROOT / 'prompts' / 'system_prompt.txt'


def _workspace_checkpoint_name(path: Path) -> str:
    """Convert a workspace path into a stable checkpoint folder name."""
    raw = str(path)
    if len(raw) >= 2 and raw[1] == ':':
        raw = raw[0].casefold() + raw[1:]
    raw = raw.replace(':', '-').replace('\\', '-').replace('/', '-').replace('_', '-')
    name = re.sub(r'[^A-Za-z0-9.-]', '-', raw).strip('-')
    return name or 'workspace'


WORKSPACE_CHECKPOINT_NAME = _workspace_checkpoint_name(WORKSPACE_ROOT)
CHECKPOINT_DIR = CHECKPOINT_ROOT / WORKSPACE_CHECKPOINT_NAME
CHECKPOINT_DB_PATH = CHECKPOINT_DIR / 'agent_checkpoints.db'
