# 3D Model Generation Pipeline

Convert 50–100 images into a `.glb` 3D model — fully automated, no API needed.

## Pipeline Flow

```
Images → COLMAP (SfM + MVS) → Dense Point Cloud → Open3D Mesh → GLB ✅
```

## Prerequisites

### 1. Python Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

*(Note: COLMAP is already bundled locally inside the `colmap/` folder. You do NOT need to download it or add it to your system PATH!)*

---

## Usage

### Basic run (uses `./images` folder by default)
```bash
python pipeline_runner.py --images ./my_photos
```

### Specify custom output folder
```bash
python pipeline_runner.py --images ./my_photos --output ./results
```

### Force CPU mode (if GPU causes issues)
```bash
python pipeline_runner.py --images ./my_photos --cpu-only
```

### Skip COLMAP (use existing point cloud)
```bash
python pipeline_runner.py --skip-colmap --ply ./workspace/dense/fused.ply
```

---

## Project Structure

```
3d_pipeline/
├── config.py               ← Edit paths and settings here
├── pipeline_runner.py      ← Main entry point
├── requirements.txt
│
├── core/
│   ├── colmap_runner.py    ← COLMAP CLI wrapper
│   ├── mesh_builder.py     ← Open3D Poisson reconstruction
│   └── converter.py        ← Trimesh GLB export
│
└── utils/
    ├── logger.py           ← Colored logging
    ├── validators.py       ← Image validation
    └── glb_validator.py    ← GLB file verification
```

---

## Tips for Best Results 📸

| Do ✅ | Avoid ❌ |
|---|---|
| 60-70% overlap between shots | Large gaps between shots |
| Consistent, diffuse lighting | Mixed lighting / shadows |
| Sharp, in-focus photos | Blurry or motion-blurred images |
| Varied angles (360° coverage) | All photos from same angle |
| Matte surface objects | Shiny / transparent objects |

---

## Output

The final model is saved to:
```
./output/model.glb
```

Open with:
- **Blender** (free, full editing)
- **Windows 3D Viewer** (built-in)
- **https://gltf.report** (online validator)
- **https://modelviewer.dev** (browser preview)
