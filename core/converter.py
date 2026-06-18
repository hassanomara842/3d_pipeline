"""
core/converter.py
-----------------
Converts the final mesh (.ply or .obj) to .glb using the trimesh library.

Why trimesh?
  - Lightweight and pure Python
  - Handles PLY/OBJ/STL → GLB conversion reliably
  - Preserves textures and vertex colors when present
  - GLB output is compact (binary glTF) and widely compatible
"""

import trimesh
import numpy as np
from pathlib import Path

from utils.logger import get_logger
from utils.glb_validator import validate_glb
import config

logger = get_logger("converter")


def load_mesh_for_export(mesh_path: str) -> trimesh.Trimesh:
    """
    Loads a mesh file (PLY or OBJ) using trimesh.

    Args:
        mesh_path: Path to the input mesh file.

    Returns:
        trimesh.Trimesh object ready for export.

    Raises:
        FileNotFoundError: If mesh file doesn't exist.
        ValueError: If the mesh is empty after loading.
    """
    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    logger.info(f"Loading mesh for export: {mesh_path}")

    # force=mesh ensures we always get a single Trimesh (not a Scene)
    mesh = trimesh.load(str(path), force="mesh")

    if mesh.is_empty:
        raise ValueError(
            "Loaded mesh is empty. The reconstruction may have produced no geometry."
        )

    logger.info(
        f"Mesh loaded — vertices: {len(mesh.vertices):,}, "
        f"faces: {len(mesh.faces):,}"
    )
    return mesh


def apply_vertex_colors(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Ensures the mesh has valid vertex colors.
    If no colors exist, applies a neutral gray so the GLB renders correctly
    in viewers instead of showing a black model.

    Args:
        mesh: Input trimesh object.

    Returns:
        Mesh with vertex colors guaranteed.
    """
    if mesh.visual is None or not hasattr(mesh.visual, "vertex_colors"):
        logger.info("No vertex colors found — applying default gray color.")
        mesh.visual.vertex_colors = np.full(
            (len(mesh.vertices), 4),
            [180, 180, 180, 255],  # RGBA gray
            dtype=np.uint8,
        )
    else:
        logger.info("Vertex colors present ✅")
    return mesh


def export_to_glb(mesh: trimesh.Trimesh, output_path: str) -> str:
    """
    Exports a trimesh object as a binary .glb file.

    Args:
        mesh: The mesh to export.
        output_path: Destination path for the .glb file.

    Returns:
        Path to the exported .glb file.

    Raises:
        RuntimeError: If the export fails.
    """
    from pathlib import Path as P
    P(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting mesh as GLB → {output_path}")

    try:
        # trimesh wraps the mesh in a Scene for proper glTF export
        scene = trimesh.scene.Scene(mesh)
        glb_bytes = scene.export(file_type="glb")

        with open(output_path, "wb") as f:
            f.write(glb_bytes)

    except Exception as e:
        raise RuntimeError(f"GLB export failed: {e}") from e

    logger.info(f"GLB exported ✅")
    return output_path


def convert_to_glb(mesh_path: str) -> str:
    """
    Full conversion pipeline: mesh file → validated .glb.

    Args:
        mesh_path: Path to the input mesh (.ply or .obj).

    Returns:
        Path to the final .glb file.
    """
    mesh = load_mesh_for_export(mesh_path)
    mesh = apply_vertex_colors(mesh)
    output_path = export_to_glb(mesh, config.OUTPUT_GLB_PATH)

    # Final validation — checks magic bytes and file size
    validate_glb(output_path)

    logger.info(f"Conversion complete ✅ → {output_path}")
    return output_path
