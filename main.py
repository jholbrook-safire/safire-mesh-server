"""
Safire Mesh Processing Server

FastAPI server providing mesh repair, orientation, and slicing.
Uses trimesh for mesh operations and SuperSlicer CLI for slicing.
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
Updated At (ISO): 2026-02-05T15:45:00Z
Updated At (PT): 2026-02-05T07:45:00 PST
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

SUPERSLICER_PATH = os.getenv("SUPERSLICER_PATH", "superslicer")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/safire-mesh"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify SuperSlicer is available
    try:
        result = subprocess.run(
            [SUPERSLICER_PATH, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"SuperSlicer available: {SUPERSLICER_PATH}")
    except FileNotFoundError:
        print("WARNING: SuperSlicer not found at", SUPERSLICER_PATH)
        print("Slicing endpoint will not be available")
    except Exception as e:
        print(f"WARNING: Could not check SuperSlicer: {e}")
    
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
    description="Mesh processing API with trimesh and SuperSlicer",
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


def lay_flat(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Simple lay-flat: just move the mesh so its lowest point is at Z=0.
    Preserves original orientation.
    """
    result = mesh.copy()
    result.vertices[:, 2] -= result.bounds[0, 2]
    return result


def auto_orient_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Orient mesh for optimal 3D printing using PrusaSlicer's algorithm.
    
    Port of PrusaSlicer's Rotfinder.cpp algorithm:
    - Tests each convex hull face as a potential base
    - Rewards floor contact (negative score)
    - Penalizes overhangs (positive score, angle³ weighting)
    - Picks orientation with minimum total score
    
    Returns:
        Oriented mesh sitting on Z=0, optimized for printing
    """
    try:
        print(f"Auto-orienting mesh with {len(mesh.faces)} faces (PrusaSlicer algorithm)...")
        
        # Get convex hull - each face is a candidate base orientation
        hull = mesh.convex_hull
        
        # Collect unique rotations from hull faces
        DOWN = np.array([0, 0, -1])
        candidates = []
        
        for i, normal in enumerate(hull.face_normals):
            area = hull.area_faces[i]
            
            # Skip tiny faces
            if area < 1e-6:
                continue
            
            # Calculate rotation to put this face down
            dot = np.dot(normal, DOWN)
            
            if dot > 0.999:
                # Already pointing down
                rotation = np.eye(4)
            elif dot < -0.999:
                # Pointing up - rotate 180 around X
                rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            else:
                axis = np.cross(normal, DOWN)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(dot, -1, 1))
                rotation = trimesh.transformations.rotation_matrix(angle, axis)
            
            candidates.append((rotation, area))
        
        # Remove near-duplicate rotations, keep ones with larger face areas
        unique_candidates = []
        for rot, area in sorted(candidates, key=lambda x: -x[1]):  # Sort by area desc
            is_duplicate = False
            for existing_rot, _ in unique_candidates:
                if np.allclose(rot, existing_rot, atol=0.1):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append((rot, area))
        
        print(f"Testing {len(unique_candidates)} candidate orientations...")
        
        # Score each candidate orientation
        best_score = float('inf')
        best_rotation = np.eye(4)
        
        for rotation, _ in unique_candidates:
            score = _score_orientation(mesh, rotation)
            if score < best_score:
                best_score = score
                best_rotation = rotation
        
        print(f"Best orientation score: {best_score:.2f}")
        
        # Apply best rotation
        result = mesh.copy()
        result.apply_transform(best_rotation)
        
        # Move to Z=0
        result.vertices[:, 2] -= result.bounds[0, 2]
        
        print(f"Final bounds: {result.bounds.tolist()}")
        return result
        
    except Exception as e:
        print(f"Auto-orient failed: {e}, using lay-flat")
        import traceback
        traceback.print_exc()
        return lay_flat(mesh)


def _score_orientation(mesh: trimesh.Trimesh, rotation: np.ndarray) -> float:
    """
    Score an orientation using PrusaSlicer's algorithm (vectorized for speed).
    
    Lower score = better orientation.
    - Floor contact: -2 * area (reward)
    - Overhangs: sqrt(area) * angle³ (penalty)
    """
    # Get rotation matrix (3x3) from 4x4 transform
    rot_matrix = rotation[:3, :3]
    
    # Rotate normals (vectorized)
    rotated_normals = mesh.face_normals @ rot_matrix.T
    
    # Rotate vertices to find Z positions
    rotated_verts = mesh.vertices @ rot_matrix.T
    z_min = rotated_verts[:, 2].min()
    rotated_verts[:, 2] -= z_min
    
    # Get Z coordinates of each face's vertices
    face_z = rotated_verts[mesh.faces, 2]  # Shape: (n_faces, 3)
    max_face_z = face_z.max(axis=1)  # Highest vertex Z for each face
    
    # Floor threshold
    z_height = rotated_verts[:, 2].max()
    z_floor = max(0.1 * z_height, 0.5)
    
    # Identify floor faces (all vertices near Z=0)
    on_floor = max_face_z <= z_floor
    
    # Get face areas
    areas = mesh.area_faces
    
    # Floor contact score: -2 * area (reward)
    floor_score = -2.0 * areas[on_floor].sum()
    
    # Overhang score for non-floor faces
    DOWN = np.array([0, 0, -1])
    cos_angles = rotated_normals @ DOWN  # Dot product with DOWN
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    phi = angles / np.pi  # Normalize to 0-1
    penalties = phi ** 3  # Cube to penalize steep overhangs
    
    # Weight by sqrt of area, only for non-floor faces
    overhang_scores = np.sqrt(areas) * penalties
    overhang_score = overhang_scores[~on_floor].sum()
    
    return floor_score + overhang_score


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
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        
        # Remove degenerate faces (zero area)
        mesh.update_faces(mesh.nondegenerate_faces())
        
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
        oriented = auto_orient_mesh(mesh)
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
    Full mesh preparation: repair + auto-orient + center + optional scale.
    
    Uses PrusaSlicer's orientation algorithm to find optimal print orientation
    that maximizes bed contact and minimizes overhangs.
    
    Args:
        file: STL/OBJ file to prepare
        target_size_mm: Scale to this max dimension (0 = no scaling)
    
    Returns prepared STL file ready for slicing.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "prepared.stl"
        
        # Use SuperSlicer for repair if available, fallback to trimesh
        repaired_path = job_dir / "repaired.stl"
        
        try:
            # Try SuperSlicer repair (better quality)
            cmd = [
                SUPERSLICER_PATH,
                "--repair",
                "--export-stl",
                "--center", "0,0",
                "--output", str(repaired_path),
                str(input_path),
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            )
            
            if repaired_path.exists():
                mesh = load_mesh(repaired_path)
            else:
                # SuperSlicer failed, use trimesh
                mesh = load_mesh(input_path)
                mesh.fill_holes()
                mesh.fix_normals()
                mesh.merge_vertices()
        except Exception:
            # Fallback to trimesh repair
            mesh = load_mesh(input_path)
            mesh.fill_holes()
            mesh.fix_normals()
            mesh.merge_vertices()
        
        # Scale if requested
        if target_size_mm > 0:
            current_size = mesh.bounds[1] - mesh.bounds[0]
            current_max = max(current_size)
            if current_max > 0:
                scale_factor = target_size_mm / current_max
                mesh.apply_scale(scale_factor)
        
        # Auto-orient for optimal printing
        mesh = auto_orient_mesh(mesh)
        
        # Center on XY origin
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
    Slice mesh to GCode using SuperSlicer.
    Returns GCode file.
    
    Note: Requires SuperSlicer to be installed.
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
        
        # Run SuperSlicer CLI
        # SuperSlicer uses PrusaSlicer-compatible CLI
        cmd = [
            SUPERSLICER_PATH,
            "--export-gcode",
            "--output", str(output_path),
            "--layer-height", str(layer_height),
            "--fill-density", f"{infill_percent}%",
            "--nozzle-diameter", "0.4",
            "--filament-diameter", "1.75",
            "--temperature", "200",
            "--bed-temperature", "60",
            "--center", "100,100",
            str(prepared_path),
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
            raise HTTPException(503, "SuperSlicer not available on this server")
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
