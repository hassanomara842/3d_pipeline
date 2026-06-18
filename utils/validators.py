"""
utils/validators.py
-------------------
Validates the image dataset before starting the pipeline.
Checks count, format, and basic readability.
"""

import os
from pathlib import Path
from typing import List

from utils.logger import get_logger
import config

logger = get_logger("validators")


def get_image_files(images_dir: str) -> List[Path]:
    """
    Scans a directory and returns all supported image files.

    Args:
        images_dir: Path to the folder containing images.

    Returns:
        List of Path objects for each valid image file.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
    """
    folder = Path(images_dir)

    if not folder.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]

    return sorted(images)


def validate_images(images_dir: str) -> List[Path]:
    """
    Full validation of the image dataset.

    Checks:
    - Directory exists
    - At least MIN_IMAGES images found
    - No more than MAX_IMAGES images
    - All files are readable

    Args:
        images_dir: Path to the folder containing images.

    Returns:
        List of valid image paths.

    Raises:
        FileNotFoundError: Directory missing.
        ValueError: Wrong number of images or unreadable files.
    """
    logger.info(f"Scanning images in: {images_dir}")
    images = get_image_files(images_dir)

    logger.info(f"Found {len(images)} image(s)")

    # ── Check count ────────────────────────────────────────────────
    if len(images) < config.MIN_IMAGES:
        raise ValueError(
            f"Too few images: found {len(images)}, minimum is {config.MIN_IMAGES}. "
            "Add more images for better reconstruction."
        )

    if len(images) > config.MAX_IMAGES:
        raise ValueError(
            f"Too many images: found {len(images)}, maximum is {config.MAX_IMAGES}. "
            "Remove some images or increase MAX_IMAGES in config.py."
        )

    # ── Check readability ──────────────────────────────────────────
    unreadable = []
    for img in images:
        try:
            with open(img, "rb") as f:
                f.read(16)  # Read first 16 bytes to check file is accessible
        except OSError:
            unreadable.append(img.name)

    if unreadable:
        raise ValueError(
            f"The following images could not be read: {unreadable}"
        )

    logger.info(f"All {len(images)} images validated successfully ✅")
    return images
