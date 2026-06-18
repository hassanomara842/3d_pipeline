"""
utils/logger.py
---------------
Colored, timestamped logger used across the entire pipeline.
"""

import logging
import sys

# ANSI color codes
COLORS = {
    "DEBUG":    "\033[94m",   # Blue
    "INFO":     "\033[92m",   # Green
    "WARNING":  "\033[93m",   # Yellow
    "ERROR":    "\033[91m",   # Red
    "CRITICAL": "\033[95m",   # Magenta
    "RESET":    "\033[0m",
}


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors and icons to log messages."""

    ICONS = {
        "DEBUG":    "🔵",
        "INFO":     "✅",
        "WARNING":  "⚠️ ",
        "ERROR":    "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        icon  = self.ICONS.get(record.levelname, "")
        reset = COLORS["RESET"]
        record.msg = f"{color}{icon} {record.msg}{reset}"
        return super().format(record)


def get_logger(name: str = "3d_pipeline") -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger name (usually the module name).

    Returns:
        logging.Logger: Configured logger with colored output.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = ColorFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
