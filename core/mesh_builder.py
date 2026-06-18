"""
core/mesh_builder.py
--------------------
Converts a dense point cloud (.ply) from COLMAP into a clean 3D mesh (.ply).

Uses Open3D's Poisson Surface Reconstruction:
  - Estimates surface normals from the point cloud
  - Runs Poisson reconstruction to build a watertight mesh
  - Removes low-density outlier vertices (cleans edges and noise)
  - Fills holes for a more complete mesh
"""

import numpy as np
import open3d as o3d
from pathlib import Path

from utils.logger import get_logger
import config

logger = get_logger("mesh_builder")


def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    """
    Loads and validates a .ply point cloud file.

    Args:
        ply_path: Path to the fused.ply from COLMAP.

    Returns:
        Open3D PointCloud object.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If the point cloud is empty.
    """
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Point cloud not found: {ply_path}\n"
            "Make sure the COLMAP dense reconstruction step completed successfully."
        )

    logger.info(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(path))

    n_points = len(pcd.points)
    if n_points == 0:
        raise ValueError("Point cloud is empty. COLMAP reconstruction may have failed.")

    logger.info(f"Loaded {n_points:,} points")
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Cleans and prepares the point cloud for meshing.

    Steps:
      1. Statistical outlier removal (removes noise points)
      2. Normal estimation (required for Poisson)

    Args:
        pcd: Raw input point cloud.

    Returns:
        Cleaned point cloud with estimated normals.
    """
    logger.info("Removing statistical outliers...")
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )
    removed = len(pcd.points) - len(pcd_clean.points)
    logger.info(f"Removed {removed:,} outlier points")

    logger.info("Estimating surface normals...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    # Orient normals consistently (important for Poisson quality)
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)
    logger.info("Normals estimated ✅")

    return pcd_clean


def run_poisson_reconstruction(
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """
    Runs Poisson Surface Reconstruction to build a mesh from the point cloud.

    Args:
        pcd: Preprocessed point cloud with normals.

    Returns:
        Reconstructed mesh with low-density vertices removed.
    """
    logger.info(f"Running Poisson reconstruction (depth={config.POISSON_DEPTH})...")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=config.POISSON_DEPTH,
        width=0,
        scale=1.1,
        linear_fit=False,
    )

    n_before = len(mesh.vertices)
    logger.info(f"Poisson mesh: {n_before:,} vertices")

    # ── Remove low-density vertices (noisy boundary artifacts) ─────
    logger.info(
        f"Removing bottom {config.DENSITY_QUANTILE*100:.0f}% density vertices (edge cleanup)..."
    )
    densities_np = np.asarray(densities)
    threshold = np.quantile(densities_np, config.DENSITY_QUANTILE)
    remove_mask = densities_np < threshold
    mesh.remove_vertices_by_mask(remove_mask)

    n_after = len(mesh.vertices)
    logger.info(f"Vertices after cleanup: {n_after:,} (removed {n_before - n_after:,})")

    return mesh


def fill_holes(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Attempts to fill small holes in the mesh for a more complete model.

    Args:
        mesh: Input mesh (may have holes from missing data).

    Returns:
        Mesh after cleanup operations.
    """
    logger.info("Running mesh cleanup (remove duplicates, degenerate triangles)...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    logger.info("Removing floating artifacts (keeping largest connected component)...")
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        logger.info(f"Removed {len(cluster_n_triangles) - 1} floating artifacts.")

    # Compute vertex normals for better visual quality
    mesh.compute_vertex_normals()
    logger.info("Mesh cleanup done ✅")
    return mesh


def save_mesh(mesh: o3d.geometry.TriangleMesh, output_path: str) -> None:
    """
    Saves the mesh as a .ply file.

    Args:
        mesh: The final cleaned mesh.
        output_path: Where to save the .ply file.
    """
    o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
    logger.info(f"Mesh saved to: {output_path}")


def build_mesh(ply_path: str) -> str:
    """
    Full mesh-building pipeline: point cloud → cleaned mesh.

    Args:
        ply_path: Path to the dense point cloud (.ply) from COLMAP.

    Returns:
        Path to the output mesh .ply file.
    """
    pcd  = load_point_cloud(ply_path)
    pcd  = preprocess_point_cloud(pcd)
    mesh = run_poisson_reconstruction(pcd)
    mesh = fill_holes(mesh)
    save_mesh(mesh, config.MESH_PLY)

    logger.info(f"Mesh building complete ✅ → {config.MESH_PLY}")
    return config.MESH_PLY
