# ============================================================
# index.py - Vercel Serverless Function Entry Point
# Located in root directory (NOT in api/ folder)
# ============================================================

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json

# Import your engine
from engine import Config, ClinicalOrchestrator

# ============================================================
# Initialize FastAPI
# ============================================================
app = FastAPI(
    title="GlucoseGuard CDSS API",
    description="Clinical Decision Support System API",
    version="1.0.0"
)

# ============================================================
# Models
# ============================================================
class PredictionRequest(BaseModel):
    cgm_readings: List[float]
    carbs: Optional[float] = 0.0
    protein: Optional[float] = 0.0
    fat: Optional[float] = 0.0
    top_k: Optional[int] = 10

class PredictionResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None

# ============================================================
# Singleton Engine Loader (lazy loading)
# ============================================================
_ENGINE = None

def get_engine():
    """Lazy load the engine - only loads on first request"""
    global _ENGINE
    if _ENGINE is None:
        try:
            print("🔄 Loading engine (first request - may take a moment)...")
            config = Config()
            _ENGINE = ClinicalOrchestrator(config)
            print("✅ Engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load engine: {str(e)}")
            raise RuntimeError(f"Failed to load engine: {str(e)}")
    return _ENGINE

# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API info"""
    return """
    <html>
        <head>
            <title>GlucoseGuard CDSS API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f7fa; }
                h1 { color: #4A90D9; }
                .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .endpoint { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4A90D9; }
                .method { color: #28a745; font-weight: bold; padding: 2px 8px; background: #e8f5e9; border-radius: 4px; }
                code { background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
                pre { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🩺 GlucoseGuard CDSS API</h1>
                <p>Clinical Decision Support System - Vercel Deployment</p>
                <p><strong>Status:</strong> 🟢 Running</p>
                
                <h2>📡 Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/predict</code>
                    <br>Predict glucose risk from CGM readings
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/health</code>
                    <br>Check API health status
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/docs</code>
                    <br>Interactive API documentation (Swagger UI)
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/redoc</code>
                    <br>ReDoc API documentation
                </div>
                
                <h2>📝 Example Request</h2>
                <pre>
POST /api/predict
Content-Type: application/json

{
    "cgm_readings": [120, 122, 118, 121, 125, 128, 130, 135, 132, 129],
    "carbs": 30,
    "protein": 10,
    "fat": 5,
    "top_k": 10
}</pre>
                
                <h2>🔗 Quick Links</h2>
                <p>
                    <a href="/docs">📚 Swagger UI</a> | 
                    <a href="/redoc">📖 ReDoc</a> | 
                    <a href="/api/health">❤️ Health Check</a>
                </p>
                
                <p style="color: #666; font-size: 0.9em; margin-top: 30px;">
                    Powered by FastAPI | Deployed on Vercel
                </p>
            </div>
        </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if engine is already loaded
        engine_status = "loaded" if _ENGINE is not None else "not_loaded"
        
        # Try to load engine (optional)
        if _ENGINE is None:
            try:
                get_engine()
                engine_status = "loaded_on_check"
            except:
                engine_status = "load_failed"
        
        return {
            "status": "healthy",
            "engine_status": engine_status,
            "message": "API is running and ready",
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "API is running but engine failed to load"
            }
        )

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict glucose risk from CGM readings
    
    Args:
        cgm_readings: List of 10 CGM readings (40-250 mg/dL)
        carbs: Carbohydrates in grams (optional)
        protein: Protein in grams (optional)
        fat: Fat in grams (optional)
        top_k: Number of food recommendations to return
    
    Returns:
        PredictionResponse with risk analysis and predictions
    """
    try:
        # Validate input
        if not request.cgm_readings or len(request.cgm_readings) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 CGM readings, got {len(request.cgm_readings)}"
            )
        
        # Validate values
        for i, val in enumerate(request.cgm_readings):
            if val < 40 or val > 250:
                raise HTTPException(
                    status_code=400,
                    detail=f"Value {val} at position {i+1} outside range (40-250 mg/dL)"
                )
        
        # Load engine (lazy)
        engine = get_engine()
        
        # Run prediction
        result = engine.run(
            cgm_readings=request.cgm_readings,
            carbs=request.carbs,
            protein=request.protein,
            fat=request.fat,
            top_k=request.top_k or 10
        )
        
        return PredictionResponse(
            status="success",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return PredictionResponse(
            status="error",
            error=str(e)
        )

# ============================================================
# Vercel Handler
# ============================================================
def handler(request, context):
    """Vercel serverless handler"""
    from mangum import Mangum
    return Mangum(app)(request, context)

# ============================================================
# For local development
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 GlucoseGuard CDSS API - Local Development")
    print("=" * 60)
    print("📊 API: http://localhost:8000/api/predict")
    print("📚 Docs: http://localhost:8000/docs")
    print("❤️ Health: http://localhost:8000/api/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
