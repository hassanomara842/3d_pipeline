"""
pipeline_runner.py
------------------
Main entry point for the 3D model generation pipeline.

Usage:
    python pipeline_runner.py --images ./my_photos
    python pipeline_runner.py --images ./my_photos --output ./results
    python pipeline_runner.py --images ./my_photos --skip-colmap --ply ./workspace/dense/fused.ply

Pipeline Flow:
    Images → COLMAP (SfM + MVS) → Dense PLY → Open3D Mesh → GLB ✅
"""

import argparse
import sys
import time
import os
import io
from pathlib import Path

# Fix Unicode output on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from utils.logger import get_logger
from utils.validators import validate_images
import config

logger = get_logger("pipeline_runner")


def print_banner() -> None:
    """Prints a startup banner."""
    banner = """
+----------------------------------------------------------+
|         [*] 3D Model Generation Pipeline [*]            |
|      Images -> COLMAP -> Open3D Mesh -> GLB Export       |
+----------------------------------------------------------+
    """
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a 3D GLB model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_runner.py --images ./photos
  python pipeline_runner.py --images ./photos --output ./my_output
  python pipeline_runner.py --skip-colmap --ply ./workspace/dense/fused.ply
        """,
    )

    parser.add_argument(
        "--images", "-i",
        type=str,
        default=config.IMAGES_DIR,
        help=f"Path to folder containing images (default: {config.IMAGES_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=config.OUTPUT_DIR,
        help=f"Output directory for the .glb file (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Skip COLMAP and use an existing .ply file (requires --ply)",
    )
    parser.add_argument(
        "--skip-to-dense",
        action="store_true",
        help="Skip SfM, resume from dense MVS",
    )
    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Path to existing dense point cloud .ply (used with --skip-colmap)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU mode for COLMAP (disables GPU/CUDA)",
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> str:
    """
    Orchestrates the full pipeline based on parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Path to the final .glb file.
    """
    start_time = time.time()

    # ── Apply CLI overrides to config ─────────────────────────────
    config.IMAGES_DIR = args.images
    config.OUTPUT_DIR = args.output
    config.OUTPUT_GLB_PATH = os.path.join(args.output, config.OUTPUT_GLB_NAME)

    if args.cpu_only:
        logger.warning("CPU-only mode enabled. This will be significantly slower.")
        config.USE_GPU = 0

    # ── Step 1: Validate Images ────────────────────────────────────
    if not args.skip_colmap:
        logger.info("=" * 55)
        logger.info("STEP 1/3 — Image Validation")
        logger.info("=" * 55)
        validate_images(config.IMAGES_DIR)

    # ── Step 2: COLMAP Pipeline ────────────────────────────────────
    if not args.skip_colmap:
        logger.info("=" * 55)
        logger.info("STEP 2/3 — COLMAP 3D Reconstruction")
        logger.info("=" * 55)
        from core import colmap_runner
        if args.skip_to_dense:
            logger.info("✅ Resuming from Dense Reconstruction...")
            fused_ply = colmap_runner.run_dense_reconstruction()
        else:
            fused_ply = colmap_runner.run_colmap_pipeline()
    else:
        # Use provided PLY file
        if not args.ply:
            logger.error("--skip-colmap requires --ply <path_to_fused.ply>")
            sys.exit(1)
        fused_ply = args.ply
        logger.info(f"Skipping COLMAP — using existing PLY: {fused_ply}")

    # ── Step 3: Build Mesh ─────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3a/3 — Mesh Building (Open3D Poisson)")
    logger.info("=" * 55)
    from core.mesh_builder import build_mesh
    mesh_ply = build_mesh(fused_ply)

    # ── Step 4: Convert to GLB ─────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3b/3 — GLB Export (trimesh)")
    logger.info("=" * 55)
    from core.converter import convert_to_glb
    glb_path = convert_to_glb(mesh_ply)

    # ── Done ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    logger.info("=" * 55)
    logger.info(f"🎉 PIPELINE COMPLETE in {minutes}m {seconds}s")
    logger.info(f"📦 Output GLB: {Path(glb_path).resolve()}")
    logger.info("=" * 55)

    return glb_path


def main() -> None:
    print_banner()
    args = parse_args()

    try:
        glb_path = run_pipeline(args)
        print(f"\n✅ Success! Your 3D model is ready: {Path(glb_path).resolve()}")
        print("   Open it in Blender, Windows 3D Viewer, or https://gltf.report\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    except RuntimeError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
