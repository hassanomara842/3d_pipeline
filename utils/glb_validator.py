"""
utils/glb_validator.py
----------------------
Validates that the exported .glb file is a real, non-empty GLB binary.
GLB files start with the magic bytes: 0x46546C67 ("glTF" in ASCII).
"""

from pathlib import Path
from utils.logger import get_logger

logger = get_logger("glb_validator")

# GLB binary magic bytes (first 4 bytes of any valid .glb file)
GLB_MAGIC = b"glTF"
MIN_GLB_SIZE_BYTES = 100  # Any valid GLB must be at least 100 bytes


def validate_glb(glb_path: str) -> bool:
    """
    Validates a .glb file by checking magic bytes and file size.

    Args:
        glb_path: Path to the .glb file.

    Returns:
        True if file is valid.

    Raises:
        FileNotFoundError: File doesn't exist.
        ValueError: File is too small or has wrong magic bytes.
    """
    path = Path(glb_path)

    # ── Existence check ────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    # ── Size check ─────────────────────────────────────────────────
    size = path.stat().st_size
    if size < MIN_GLB_SIZE_BYTES:
        raise ValueError(
            f"GLB file is too small ({size} bytes). "
            "The export may have failed or produced an empty model."
        )

    # ── Magic bytes check ──────────────────────────────────────────
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic != GLB_MAGIC:
        raise ValueError(
            f"File does not appear to be a valid GLB. "
            f"Expected magic bytes {GLB_MAGIC!r}, got {magic!r}."
        )

    size_mb = size / (1024 * 1024)
    logger.info(f"GLB validation passed — file size: {size_mb:.2f} MB")
    return True
