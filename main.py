"""
Safire Mesh Processing Server

FastAPI server providing mesh repair, orientation, and slicing.
Uses trimesh for mesh operations and CuraEngine for slicing.
Designed to run as a Docker container on Railway, Render, or similar platforms.

Endpoints:
  POST /repair     - Repair mesh issues, return fixed STL
  POST /orient     - Auto-orient for printing, return oriented STL  
  POST /prepare    - Full preparation (repair + center + orient + scale)
  POST /slice      - Slice to GCode with print settings
  POST /analyze    - Analyze mesh and return statistics
  POST /simplify   - Simplify mesh to reduce polygon count
  GET  /health     - Health check

Created At (ISO): 2026-02-04T00:00:00Z
Created At (PT): 2026-02-03T16:00:00 PST
Updated At (ISO): 2026-02-05T14:00:00Z
Updated At (PT): 2026-02-05T06:00:00 PST
Updated By: AI
"""

import os
import uuid
import asyncio
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import trimesh
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================================
# Configuration
# ============================================================================

CURA_ENGINE_PATH = os.getenv("CURA_ENGINE_PATH", "CuraEngine")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/safire-mesh"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify CuraEngine is available
    try:
        result = subprocess.run(
            [CURA_ENGINE_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"CuraEngine: {result.stdout.strip() or result.stderr.strip()}")
    except FileNotFoundError:
        print("WARNING: CuraEngine not found at", CURA_ENGINE_PATH)
        print("Slicing endpoint will not be available")
    except Exception as e:
        print(f"WARNING: Could not check CuraEngine: {e}")
    
    yield
    
    # Shutdown: cleanup temp files
    if TEMP_DIR.exists():
        for item in TEMP_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)


# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="Safire Mesh Server",
    description="Mesh processing API with trimesh and CuraEngine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================

class MeshStats(BaseModel):
    vertices: int
    faces: int
    is_watertight: bool
    is_manifold: bool
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    bounding_box: dict
    center_of_mass: Optional[list] = None


# ============================================================================
# Utilities
# ============================================================================

def create_job_dir() -> Path:
    """Create a unique temporary directory for a job."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


async def save_upload(file: UploadFile, job_dir: Path) -> Path:
    """Save uploaded file to job directory."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")
    
    # Determine file extension from filename
    ext = Path(file.filename or "model.stl").suffix.lower() or ".stl"
    input_path = job_dir / f"input{ext}"
    
    with open(input_path, "wb") as f:
        f.write(content)
    
    return input_path


def cleanup_job(job_dir: Path):
    """Clean up job directory."""
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except:
        pass


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh file and ensure it's a Trimesh."""
    mesh = trimesh.load(str(path))
    
    # Handle scene (multiple meshes) by concatenating
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise HTTPException(400, "No valid geometry found in file")
        mesh = trimesh.util.concatenate(meshes)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise HTTPException(400, "Could not load as triangle mesh")
    
    return mesh


def find_optimal_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Find optimal print orientation by maximizing flat surface area on build plate.
    Uses convex hull and analyzes face normals.
    """
    # Get convex hull for stability analysis
    hull = mesh.convex_hull
    
    # Find faces that could be the "bottom" (facing -Z)
    best_score = -1
    best_transform = np.eye(4)
    
    # Test each face of the convex hull as potential bottom
    for i, normal in enumerate(hull.face_normals):
        # Calculate rotation to align this face normal with -Z
        target = np.array([0, 0, -1])
        
        # Skip if already aligned
        dot = np.dot(normal, target)
        if abs(dot) > 0.999:
            rotation = np.eye(4)
        else:
            # Rodrigues rotation formula
            axis = np.cross(normal, target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                angle = np.arccos(np.clip(dot, -1, 1))
                rotation = trimesh.transformations.rotation_matrix(angle, axis)
            else:
                rotation = np.eye(4)
        
        # Score based on face area (larger flat bottom = better)
        face_area = hull.area_faces[i]
        
        # Bonus for faces closer to horizontal
        flatness = abs(dot)
        score = face_area * flatness
        
        if score > best_score:
            best_score = score
            best_transform = rotation
    
    # Apply the best rotation
    oriented = mesh.copy()
    oriented.apply_transform(best_transform)
    
    # Move to sit on Z=0 plane
    oriented.vertices[:, 2] -= oriented.bounds[0, 2]
    
    return oriented


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "safire-mesh-server"}


@app.post("/analyze")
async def analyze_mesh(file: UploadFile = File(...)):
    """
    Analyze mesh and return statistics.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        mesh = load_mesh(input_path)
        
        bounds = mesh.bounds
        stats = MeshStats(
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            is_watertight=mesh.is_watertight,
            is_manifold=mesh.is_manifold if hasattr(mesh, 'is_manifold') else False,
            volume=float(mesh.volume) if mesh.is_watertight else None,
            surface_area=float(mesh.area),
            bounding_box={
                "min": bounds[0].tolist(),
                "max": bounds[1].tolist(),
                "size": (bounds[1] - bounds[0]).tolist(),
            },
            center_of_mass=mesh.center_mass.tolist() if mesh.is_watertight else None,
        )
        
        return stats
        
    finally:
        cleanup_job(job_dir)


@app.post("/repair")
async def repair_mesh(file: UploadFile = File(...)):
    """
    Repair mesh issues (holes, normals, duplicates).
    Returns repaired STL file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "repaired.stl"
        
        mesh = load_mesh(input_path)
        
        # Repair operations
        mesh.fill_holes()
        mesh.fix_normals()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Merge close vertices
        mesh.merge_vertices()
        
        mesh.export(str(output_path))
        
        return FileResponse(
            output_path,
            media_type="model/stl",
            filename="repaired.stl",
        )
        
    except HTTPException:
        cleanup_job(job_dir)
        raise
    except Exception as e:
        cleanup_job(job_dir)
        raise HTTPException(500, f"Repair failed: {str(e)}")


@app.post("/orient")
async def orient_mesh(file: UploadFile = File(...)):
    """
    Auto-orient mesh for optimal printing (maximize flat bottom).
    Returns oriented STL file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "oriented.stl"
        
        mesh = load_mesh(input_path)
        oriented = find_optimal_orientation(mesh)
        oriented.export(str(output_path))
        
        return FileResponse(
            output_path,
            media_type="model/stl",
            filename="oriented.stl",
        )
        
    except HTTPException:
        cleanup_job(job_dir)
        raise
    except Exception as e:
        cleanup_job(job_dir)
        raise HTTPException(500, f"Orient failed: {str(e)}")


@app.post("/prepare")
async def prepare_mesh(
    file: UploadFile = File(...),
    target_size_mm: float = Form(default=0),
):
    """
    Full mesh preparation: repair + center + orient + optional scale.
    Returns prepared STL file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "prepared.stl"
        
        mesh = load_mesh(input_path)
        
        # 1. Repair
        mesh.fill_holes()
        mesh.fix_normals()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.merge_vertices()
        
        # 2. Scale if requested
        if target_size_mm > 0:
            current_size = mesh.bounds[1] - mesh.bounds[0]
            current_max = max(current_size)
            if current_max > 0:
                scale_factor = target_size_mm / current_max
                mesh.apply_scale(scale_factor)
        
        # 3. Orient for printing
        mesh = find_optimal_orientation(mesh)
        
        # 4. Center on origin (XY)
        centroid = mesh.centroid
        mesh.vertices[:, 0] -= centroid[0]
        mesh.vertices[:, 1] -= centroid[1]
        
        mesh.export(str(output_path))
        
        return FileResponse(
            output_path,
            media_type="model/stl",
            filename="prepared.stl",
        )
        
    except HTTPException:
        cleanup_job(job_dir)
        raise
    except Exception as e:
        cleanup_job(job_dir)
        raise HTTPException(500, f"Preparation failed: {str(e)}")


@app.post("/slice")
async def slice_mesh(
    file: UploadFile = File(...),
    layer_height: float = Form(default=0.2),
    infill_percent: int = Form(default=20),
):
    """
    Slice mesh to GCode using CuraEngine.
    Returns GCode file.
    
    Note: Requires CuraEngine to be installed.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "output.gcode"
        
        # First prepare the mesh
        mesh = load_mesh(input_path)
        mesh.fill_holes()
        mesh.fix_normals()
        prepared_path = job_dir / "prepared.stl"
        mesh.export(str(prepared_path))
        
        # Run CuraEngine
        # Note: CuraEngine requires a definition file for settings
        # This is a simplified example - production would need proper config
        cmd = [
            CURA_ENGINE_PATH,
            "slice",
            "-j", "/usr/share/cura/resources/definitions/fdmprinter.def.json",
            "-o", str(output_path),
            "-s", f"layer_height={layer_height}",
            "-s", f"infill_sparse_density={infill_percent}",
            "-l", str(prepared_path),
        ]
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            )
            
            if not output_path.exists():
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise HTTPException(500, f"Slicing failed: {error_msg}")
                
        except FileNotFoundError:
            raise HTTPException(503, "CuraEngine not available on this server")
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "Slicing timed out")
        
        return FileResponse(
            output_path,
            media_type="text/x-gcode",
            filename="print.gcode",
        )
        
    except HTTPException:
        cleanup_job(job_dir)
        raise
    except Exception as e:
        cleanup_job(job_dir)
        raise HTTPException(500, f"Slicing failed: {str(e)}")


@app.post("/simplify")
async def simplify_mesh(
    file: UploadFile = File(...),
    target_faces: int = Form(default=10000),
):
    """
    Simplify mesh to reduce polygon count.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "simplified.stl"
        
        mesh = load_mesh(input_path)
        
        if len(mesh.faces) > target_faces:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            simplified.export(str(output_path))
        else:
            mesh.export(str(output_path))
        
        return FileResponse(
            output_path,
            media_type="model/stl",
            filename="simplified.stl",
        )
        
    except HTTPException:
        cleanup_job(job_dir)
        raise
    except Exception as e:
        cleanup_job(job_dir)
        raise HTTPException(500, f"Simplification failed: {str(e)}")


# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
