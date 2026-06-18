"""
config.py
---------
Central configuration for the 3D pipeline.
Edit these paths before running the pipeline.
"""

import os

# ─── Input / Output ───────────────────────────────────────────────
IMAGES_DIR   = "./images"          # Folder containing your 50-100 photos
OUTPUT_DIR   = "./output"          # Where the final .glb will be saved
WORK_DIR     = "./workspace"       # Temp working directory (COLMAP files)

# ─── COLMAP Settings ──────────────────────────────────────────────
# Path to COLMAP executable. If added to PATH, leave as "colmap"
COLMAP_BIN   = r"C:\Users\lenovo\.gemini\antigravity-ide\scratch\3d_pipeline\colmap\bin\colmap.exe"

# Use GPU for COLMAP steps (1 = yes, 0 = CPU only)
# MX450 supports CUDA, so we try GPU first
USE_GPU      = 1

# Camera model — SIMPLE_RADIAL works well for phone cameras
CAMERA_MODEL = "SIMPLE_RADIAL"

# Feature matcher — "exhaustive" for <200 imgs, "sequential" for video frames
MATCHER_TYPE = "exhaustive_matcher"

# Poisson mesh depth — higher = more detail, more RAM (8-10 recommended)
POISSON_DEPTH = 9

# Quantile of low-density vertices to remove (e.g., 0.01 = remove bottom 1%)
DENSITY_QUANTILE = 0.01

# ─── Validation ───────────────────────────────────────────────────
MIN_IMAGES = 50   # أقل من كده بيضعّف الدقة بشكل واضح
MAX_IMAGES = 100  # أكتر من كده بيبطّأ جداً على الـ MX450

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

# ─── Output filename ──────────────────────────────────────────────
OUTPUT_GLB_NAME = "model.glb"

# ─── Derived Paths (don't edit) ───────────────────────────────────
DB_PATH         = os.path.join(WORK_DIR, "database.db")
SPARSE_DIR      = os.path.join(WORK_DIR, "sparse")
DENSE_DIR       = os.path.join(WORK_DIR, "dense")
FUSED_PLY       = os.path.join(DENSE_DIR, "fused.ply")
MESH_PLY        = os.path.join(WORK_DIR, "mesh.ply")
OUTPUT_GLB_PATH = os.path.join(OUTPUT_DIR, OUTPUT_GLB_NAME)
