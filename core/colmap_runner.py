"""
core/colmap_runner.py
---------------------
Wrapper around the COLMAP CLI.
Runs the full Structure-from-Motion (SfM) + Multi-View Stereo (MVS) pipeline.

Pipeline steps:
  1. feature_extractor  → detects keypoints in each image
  2. exhaustive_matcher → matches keypoints across all image pairs
  3. mapper             → builds sparse 3D point cloud (SfM)
  4. image_undistorter  → prepares images for dense MVS
  5. patch_match_stereo → dense depth estimation (GPU)
  6. stereo_fusion      → fuses depth maps into a dense point cloud (.ply)
"""

import os
import subprocess
import shutil
from pathlib import Path

from utils.logger import get_logger
import config

logger = get_logger("colmap_runner")


def _run(cmd: list, step_name: str, fallback_to_cpu: bool = False) -> None:
    """
    Executes a COLMAP command via subprocess with full error handling.

    Args:
        cmd: List of command arguments.
        step_name: Human-readable name for logging.
        fallback_to_cpu: If True and GPU fails, retry with GPU disabled.

    Raises:
        RuntimeError: If the command fails even after CPU fallback.
    """
    logger.info(f"Starting: {step_name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(result.stdout[-500:] if result.stdout else "")
        logger.info(f"Done: {step_name} ✅")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown error"

        # ── Auto-fallback to CPU if GPU failed ────────────────────
        if fallback_to_cpu and "CUDA" in error_msg.upper():
            logger.warning(
                f"GPU step failed (CUDA error). "
                f"Retrying '{step_name}' on CPU — this will be slower."
            )
            # Replace GPU flags with CPU equivalents
            cpu_cmd = [
                arg.replace("--SiftExtraction.use_gpu 1", "--SiftExtraction.use_gpu 0")
                for arg in cmd
            ]
            # Rebuild properly with use_gpu=0
        cpu_cmd = []
        skip_next = False
        for i, arg in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if arg in ("--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu",
                       "--SiftExtraction.use_gpu", "--SiftMatching.use_gpu",
                       "--PatchMatchStereo.gpu_index"):
                cpu_cmd.append(arg)
                cpu_cmd.append("0" if "index" not in arg else "-1")
                skip_next = True
            else:
                cpu_cmd.append(arg)

            try:
                subprocess.run(cpu_cmd, check=True, capture_output=True, text=True)
                logger.info(f"Done (CPU fallback): {step_name} ✅")
                return
            except subprocess.CalledProcessError as cpu_e:
                raise RuntimeError(
                    f"Step '{step_name}' failed on both GPU and CPU.\n"
                    f"Error: {cpu_e.stderr}"
                ) from cpu_e

        raise RuntimeError(
            f"Step '{step_name}' failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error: {error_msg}"
        ) from e


def prepare_workspace() -> None:
    """
    Creates all required COLMAP workspace directories.
    Cleans up any previous run to avoid stale data.
    """
    dirs = [config.WORK_DIR, config.SPARSE_DIR, config.DENSE_DIR, config.OUTPUT_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Remove old database to avoid conflicts with previous runs
    db = Path(config.DB_PATH)
    if db.exists():
        logger.warning("Found existing COLMAP database — removing to start fresh.")
        db.unlink()

    logger.info("Workspace prepared ✅")


def run_feature_extraction() -> None:
    """Step 1: Extract SIFT features from all images."""
    cmd = [
        config.COLMAP_BIN, "feature_extractor",
        "--database_path",              config.DB_PATH,
        "--image_path",                 config.IMAGES_DIR,
        "--ImageReader.camera_model",   config.CAMERA_MODEL,
        "--ImageReader.single_camera",  "1",
        "--FeatureExtraction.use_gpu",  str(config.USE_GPU),  # COLMAP 4.x: FeatureExtraction (not SiftExtraction)
    ]
    _run(cmd, "Feature Extraction", fallback_to_cpu=True)


def run_feature_matching() -> None:
    """Step 2: Match features across all image pairs."""
    cmd = [
        config.COLMAP_BIN, config.MATCHER_TYPE,
        "--database_path",           config.DB_PATH,
        "--FeatureMatching.use_gpu", str(config.USE_GPU),  # COLMAP 4.x: FeatureMatching (not SiftMatching)
    ]
    _run(cmd, "Feature Matching", fallback_to_cpu=True)


def run_sparse_reconstruction() -> None:
    """Step 3: Structure-from-Motion — builds sparse point cloud."""
    cmd = [
        config.COLMAP_BIN, "mapper",
        "--database_path",  config.DB_PATH,
        "--image_path",     config.IMAGES_DIR,
        "--output_path",    config.SPARSE_DIR,
    ]
    _run(cmd, "Sparse Reconstruction (SfM)")

    # COLMAP puts results in numbered subdirs (0, 1, 2...)
    # We expect model 0 to be the main reconstruction
    sparse_model = os.path.join(config.SPARSE_DIR, "0")
    if not os.path.exists(sparse_model):
        raise RuntimeError(
            "Sparse reconstruction produced no output. "
            "This usually means there is not enough overlap between images. "
            "Ensure images have at least 60-70% overlap."
        )


def run_dense_reconstruction() -> None:
    """
    Steps 4-6: MVS dense reconstruction.
    Undistorts images → computes depth maps → fuses into dense point cloud.
    """
    sparse_model = os.path.join(config.SPARSE_DIR, "0")

    # Step 4: Undistort images
    cmd_undistort = [
        config.COLMAP_BIN, "image_undistorter",
        "--image_path",     config.IMAGES_DIR,
        "--input_path",     sparse_model,
        "--output_path",    config.DENSE_DIR,
        "--output_type",    "COLMAP",
    ]
    _run(cmd_undistort, "Image Undistortion")

    # Step 5: Dense depth estimation (most GPU-intensive step)
    cmd_stereo = [
        config.COLMAP_BIN, "patch_match_stereo",
        "--workspace_path",                     config.DENSE_DIR,
        "--workspace_format",                   "COLMAP",
        "--PatchMatchStereo.geom_consistency",  "true",
        "--PatchMatchStereo.max_image_size",    "1000",  # تصغير جودة الصورة لتسريع المعالجة جداً وتقليل الرام
        "--PatchMatchStereo.window_step",       "2",     # تخطي بيكسلات لتسريع المعالجة
    ]
    _run(cmd_stereo, "Dense Stereo Matching (MVS)", fallback_to_cpu=True)

    # Step 6: Fuse depth maps into single dense .ply point cloud
    cmd_fuse = [
        config.COLMAP_BIN, "stereo_fusion",
        "--workspace_path",     config.DENSE_DIR,
        "--workspace_format",   "COLMAP",
        "--input_type",         "geometric",
        "--output_path",        config.FUSED_PLY,
    ]
    _run(cmd_fuse, "Stereo Fusion → fused.ply")
    return config.FUSED_PLY


def run_colmap_pipeline() -> str:
    """
    Runs the complete COLMAP pipeline end-to-end.

    Returns:
        Path to the fused dense point cloud (.ply).

    Raises:
        RuntimeError: If any COLMAP step fails.
        FileNotFoundError: If COLMAP is not installed.
    """
    # Check COLMAP is installed
    if not shutil.which(config.COLMAP_BIN):
        raise FileNotFoundError(
            f"COLMAP not found. Please install it from https://colmap.github.io "
            f"and make sure '{config.COLMAP_BIN}' is in your PATH."
        )

    prepare_workspace()
    run_feature_extraction()
    run_feature_matching()
    run_sparse_reconstruction()
    run_dense_reconstruction()

    logger.info(f"COLMAP pipeline complete. Dense point cloud: {config.FUSED_PLY}")
    return config.FUSED_PLY
