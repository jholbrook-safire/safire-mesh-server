# Safire Mesh Server

Mesh processing API server using trimesh for mesh operations and CuraEngine for slicing. Provides repair, orientation, and slicing capabilities for 3D printing workflows.

## Features

- **Mesh Analysis** - Get detailed statistics (vertices, faces, watertight, volume, etc.)
- **Mesh Repair** - Fix non-manifold geometry, holes, and other issues
- **Auto-Orient** - Optimize print orientation for minimal supports
- **Prepare** - Full pipeline: repair + center + orient + optional scaling
- **Slice** - Generate GCode with customizable settings
- **Simplify** - Reduce polygon count for complex meshes

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze mesh, return stats |
| `/repair` | POST | Repair mesh issues |
| `/orient` | POST | Auto-orient for printing |
| `/prepare` | POST | Full prep (repair + center + orient) |
| `/slice` | POST | Slice to GCode |
| `/simplify` | POST | Reduce polygon count |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install PrusaSlicer (macOS)
brew install --cask prusaslicer

# Or download from: https://www.prusa3d.com/page/prusaslicer_424/

# Run server
python main.py
# Or with uvicorn directly:
uvicorn main:app --reload --port 8000
```

## Docker

```bash
# Build
docker build -t safire-mesh-server .

# Run
docker run -p 8000:8000 safire-mesh-server

# Or with compose
docker-compose up
```

## API Usage

### Analyze Mesh

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@model.stl"
```

Response:
```json
{
  "vertices": 12000,
  "faces": 24000,
  "is_watertight": true,
  "is_manifold": true,
  "volume": 1234.56,
  "surface_area": 789.01,
  "bounding_box": {
    "min": [0, 0, 0],
    "max": [50, 50, 30],
    "size": [50, 50, 30]
  }
}
```

### Repair Mesh

```bash
curl -X POST http://localhost:8000/repair \
  -F "file=@broken-model.stl" \
  --output repaired.stl
```

### Prepare for Printing

```bash
curl -X POST http://localhost:8000/prepare \
  -F "file=@model.stl" \
  -F "target_size_mm=50" \
  --output prepared.stl
```

### Slice to GCode

```bash
curl -X POST http://localhost:8000/slice \
  -F "file=@model.stl" \
  -F "layer_height=0.2" \
  -F "infill_percent=20" \
  -F "support_material=true" \
  --output print.gcode
```

## Deployment

### Railway

1. Push to GitHub
2. Create new project in Railway
3. Connect GitHub repo
4. Railway will auto-detect Dockerfile
5. Add any needed env vars
6. Deploy!

### Render

1. Create new Web Service
2. Connect GitHub repo
3. Select Docker environment
4. Deploy

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `PRUSA_SLICER_PATH` | prusa-slicer | Path to PrusaSlicer binary |
| `TEMP_DIR` | /tmp/safire-mesh | Temp file directory |
| `MAX_FILE_SIZE_MB` | 50 | Max upload size in MB |

## Frontend Integration

Example TypeScript client:

```typescript
async function repairMesh(file: File): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('https://your-server.railway.app/repair', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Repair failed');
  }
  
  return response.blob();
}
```

## License

MIT
