from pathlib import Path
from langchain_core.tools import tool

from job_hunt.security.sandbox import validate_path

MAX_FILE_SIZE = 100 * 1024  # 100KB; truncate if exceeded
CHUNK_READ_SIZE = 4096       # bytes to read for encoding / binary detection


def _is_text_file(path: Path) -> tuple[bool, str]:
    """Check whether *path* is a text file. Returns (is_text, reason)."""
    try:
        with open(path, 'rb') as f:
            chunk = f.read(CHUNK_READ_SIZE)
    except Exception as e:
        return False, f'cannot open file: {e}'

    if not chunk:
        return True, ''

    # Null bytes almost always mean binary
    if b'\x00' in chunk:
        return False, 'null bytes detected — looks like a binary file'

    # Allow up to 30% non-printable bytes before treating as binary
    non_printable = sum(1 for b in chunk if b > 127 or (b < 32 and b not in (9, 10, 13)))
    if non_printable > len(chunk) * 0.3:
        return False, (
            f'too many non-printable bytes ({non_printable}/{len(chunk)})'
            f' — looks like a binary file'
        )

    return True, ''


def _detect_and_read(path: Path) -> str:
    """Try multiple encodings to read *path* and return its content."""
    for encoding in ('utf-8', 'gbk', 'latin-1'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            raise

    raise ValueError('all encoding attempts failed')


@tool
def read_file(file_path: str) -> str:
    """Read a local file and return its content as a string.

    Supports common text files: code (.py/.js/.ts/.java/.go etc.),
    config (.json/.yaml/.toml/.env etc.), docs (.md/.txt/.rst/.tex etc.),
    web (.html/.css/.scss etc.), data (.csv/.xml etc.).

    Args:
        file_path: path to the file, relative or absolute.

    Returns:
        File content as a string. Truncated with a summary if > MAX_FILE_SIZE.
        Returns an error message for binary files.
    """
    path, error = validate_path(file_path)
    if error:
        return f'Error: {error}'

    if not path.exists():
        return f'Error: file not found — {path}'
    if not path.is_file():
        return f'Error: not a file — {path}'

    # Check empty
    if path.stat().st_size == 0:
        return f'File is empty: {path}'

    # Check text vs binary (reads only first 4KB)
    is_text, reason = _is_text_file(path)
    if not is_text:
        return f'Error: {reason} — {path}'

    # Read the file
    try:
        content = _detect_and_read(path)
    except Exception as e:
        return f'Error: failed to read file — {path}, reason: {e}'

    # Truncate if too large
    if len(content) > MAX_FILE_SIZE:
        content = (
            f'[File too large, truncated: {len(content):,} chars, '
            f'showing first {MAX_FILE_SIZE:,} chars]\n\n'
            f'{content[:MAX_FILE_SIZE]}\n\n'
            f'[End of truncation]'
        )

    return content
