"""
Safire Mesh Processing Server

FastAPI server providing mesh repair, orientation, and slicing via PrusaSlicer CLI.
Designed to run as a Docker container on Railway, Render, or similar platforms.

Endpoints:
  POST /repair     - Repair mesh issues, return fixed STL
  POST /orient     - Auto-orient for printing, return oriented STL
  POST /prepare    - Full preparation (repair + center + orient)
  POST /slice      - Slice to GCode with print settings
  POST /analyze    - Analyze mesh and return statistics
  GET  /health     - Health check

Created: 2026-02-04
"""

import os
import uuid
import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import trimesh
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


# ============================================================================
# Configuration
# ============================================================================

PRUSA_SLICER_PATH = os.getenv("PRUSA_SLICER_PATH", "prusa-slicer")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/safire-mesh"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify PrusaSlicer is available
    try:
        result = subprocess.run(
            [PRUSA_SLICER_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"PrusaSlicer version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("WARNING: PrusaSlicer not found at", PRUSA_SLICER_PATH)
    except Exception as e:
        print(f"WARNING: Could not check PrusaSlicer: {e}")
    
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
    description="Mesh processing API with PrusaSlicer and trimesh",
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


class SliceSettings(BaseModel):
    layer_height: float = 0.2
    infill_percent: int = 20
    support_material: bool = False
    # Add more as needed


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
    # Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")
    
    # Save to temp file
    input_path = job_dir / "input.stl"
    with open(input_path, "wb") as f:
        f.write(content)
    
    return input_path


async def run_prusa_slicer(args: list, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run PrusaSlicer CLI with given arguments."""
    cmd = [PRUSA_SLICER_PATH] + args
    
    try:
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        )
        return result
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Processing timed out")
    except Exception as e:
        raise HTTPException(500, f"PrusaSlicer error: {str(e)}")


def cleanup_job(job_dir: Path):
    """Clean up job directory."""
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except:
        pass


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
    Uses trimesh for detailed analysis.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        
        # Load with trimesh for analysis
        mesh = trimesh.load(str(input_path))
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(400, "Could not load as triangle mesh")
        
        # Compute stats
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
    Repair mesh issues using PrusaSlicer.
    Returns repaired STL file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "repaired.stl"
        
        # Run PrusaSlicer repair
        result = await run_prusa_slicer([
            "--repair",
            "--export-stl", str(output_path),
            str(input_path)
        ])
        
        if not output_path.exists():
            # If PrusaSlicer didn't create output, try trimesh as fallback
            mesh = trimesh.load(str(input_path))
            if isinstance(mesh, trimesh.Trimesh):
                # Basic repair: fill holes, fix normals
                mesh.fill_holes()
                mesh.fix_normals()
                mesh.export(str(output_path))
        
        if not output_path.exists():
            raise HTTPException(500, "Repair failed to produce output")
        
        return FileResponse(
            output_path,
            media_type="model/stl",
            filename="repaired.stl",
            background=None  # Don't cleanup immediately
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
    Auto-orient mesh for optimal printing (minimize supports).
    Returns oriented STL file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "oriented.stl"
        
        # Run PrusaSlicer orient
        result = await run_prusa_slicer([
            "--orient",
            "--center",
            "--export-stl", str(output_path),
            str(input_path)
        ])
        
        if not output_path.exists():
            raise HTTPException(500, "Orient failed to produce output")
        
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
    Full mesh preparation: repair + center + orient.
    Optionally scale to target size.
    Returns prepared STL file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "prepared.stl"
        
        # Build PrusaSlicer command
        args = [
            "--repair",
            "--center",
            "--orient",
        ]
        
        # Add scaling if requested
        if target_size_mm > 0:
            # First, load to get current size
            mesh = trimesh.load(str(input_path))
            if isinstance(mesh, trimesh.Trimesh):
                current_max = max(mesh.bounds[1] - mesh.bounds[0])
                if current_max > 0:
                    scale_factor = target_size_mm / current_max
                    args.extend(["--scale", str(scale_factor)])
        
        args.extend([
            "--export-stl", str(output_path),
            str(input_path)
        ])
        
        result = await run_prusa_slicer(args)
        
        if not output_path.exists():
            raise HTTPException(500, "Preparation failed to produce output")
        
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
    support_material: bool = Form(default=False),
    printer_profile: str = Form(default=""),
):
    """
    Slice mesh to GCode using PrusaSlicer.
    Returns GCode file.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "output.gcode"
        
        # Build command
        args = [
            "--repair",
            "--center",
            "--orient",
            f"--layer-height={layer_height}",
            f"--fill-density={infill_percent}%",
        ]
        
        if support_material:
            args.append("--support-material")
        
        if printer_profile:
            args.extend(["--load", printer_profile])
        
        args.extend([
            "--export-gcode",
            "-o", str(output_path),
            str(input_path)
        ])
        
        result = await run_prusa_slicer(args, timeout=300)  # Slicing can take longer
        
        if not output_path.exists():
            # Check for errors
            error_msg = result.stderr if result else "Unknown error"
            raise HTTPException(500, f"Slicing failed: {error_msg}")
        
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
    Uses trimesh decimation.
    """
    job_dir = create_job_dir()
    
    try:
        input_path = await save_upload(file, job_dir)
        output_path = job_dir / "simplified.stl"
        
        # Load and simplify with trimesh
        mesh = trimesh.load(str(input_path))
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(400, "Could not load as triangle mesh")
        
        # Simplify
        if len(mesh.faces) > target_faces:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            simplified.export(str(output_path))
        else:
            # Already simple enough
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
