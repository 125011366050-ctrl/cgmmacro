# ============================================================
# index.py - Vercel Serverless Function (Root Directory)
# Direct deployment - NO api/ folder
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
import warnings
warnings.filterwarnings('ignore')

# Import your engine
from engine import Config, ClinicalOrchestrator

# ============================================================
# Initialize FastAPI
# ============================================================
app = FastAPI(
    title="GlucoseGuard CDSS API",
    description="Clinical Decision Support System - Vercel Deployment",
    version="1.0.0"
)

# ============================================================
# Request/Response Models
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
# Singleton Engine Loader (Lazy Loading)
# ============================================================
_ENGINE = None
_ENGINE_LOADING = False

def get_engine():
    """Lazy load the engine - only loads on first request"""
    global _ENGINE, _ENGINE_LOADING
    
    if _ENGINE is not None:
        return _ENGINE
    
    if _ENGINE_LOADING:
        raise HTTPException(
            status_code=503,
            detail="Engine is currently loading. Please try again in a few seconds."
        )
    
    try:
        _ENGINE_LOADING = True
        print("🔄 Loading CDSS engine (first request - may take a moment)...")
        
        config = Config()
        _ENGINE = ClinicalOrchestrator(config)
        
        print("✅ CDSS engine loaded successfully!")
        return _ENGINE
        
    except Exception as e:
        print(f"❌ Failed to load engine: {str(e)}")
        raise RuntimeError(f"Failed to load engine: {str(e)}")
    
    finally:
        _ENGINE_LOADING = False

# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API info and interactive UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GlucoseGuard CDSS API</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 40px 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 { 
                color: #2d3748;
                font-size: 2.5em;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .subtitle {
                color: #718096;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .status-badge {
                display: inline-block;
                background: #48bb78;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 600;
            }
            .endpoint-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 25px 0;
            }
            @media (max-width: 600px) {
                .endpoint-grid { grid-template-columns: 1fr; }
                h1 { font-size: 1.8em; }
            }
            .endpoint-card {
                background: #f7fafc;
                padding: 15px 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .endpoint-card .method {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 4px;
                font-size: 0.7em;
                font-weight: 700;
                text-transform: uppercase;
                background: #48bb78;
                color: white;
            }
            .endpoint-card .method.post { background: #4299e1; }
            .endpoint-card .method.get { background: #48bb78; }
            .endpoint-card code {
                display: block;
                margin-top: 8px;
                font-size: 0.9em;
                background: #edf2f7;
                padding: 8px 12px;
                border-radius: 6px;
                color: #2d3748;
            }
            .endpoint-card p {
                color: #718096;
                font-size: 0.9em;
                margin-top: 8px;
            }
            .example {
                background: #2d3748;
                color: #f7fafc;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                overflow-x: auto;
            }
            .example pre {
                margin: 0;
                font-size: 0.85em;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .links {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                margin-top: 25px;
                padding-top: 25px;
                border-top: 2px solid #edf2f7;
            }
            .links a {
                color: #667eea;
                text-decoration: none;
                font-weight: 500;
            }
            .links a:hover { text-decoration: underline; }
            .footer {
                margin-top: 30px;
                color: #a0aec0;
                font-size: 0.85em;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🩺 GlucoseGuard CDSS</h1>
            <div class="subtitle">
                Clinical Decision Support System 
                <span class="status-badge">● Live</span>
            </div>
            
            <p style="color: #4a5568; margin-bottom: 20px;">
                API for real-time glucose prediction, risk assessment, 
                and personalized recommendations.
            </p>
            
            <h2 style="color: #2d3748; font-size: 1.3em; margin-top: 30px;">📡 Endpoints</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card">
                    <span class="method post">POST</span>
                    <code>/api/predict</code>
                    <p>Predict glucose risk from 10 CGM readings</p>
                </div>
                <div class="endpoint-card">
                    <span class="method get">GET</span>
                    <code>/api/health</code>
                    <p>Check API and engine health status</p>
                </div>
                <div class="endpoint-card">
                    <span class="method get">GET</span>
                    <code>/docs</code>
                    <p>Interactive Swagger UI documentation</p>
                </div>
                <div class="endpoint-card">
                    <span class="method get">GET</span>
                    <code>/redoc</code>
                    <p>ReDoc API documentation</p>
                </div>
            </div>
            
            <h2 style="color: #2d3748; font-size: 1.3em; margin-top: 30px;">📝 Example Request</h2>
            <div class="example">
                <pre>
curl -X POST https://your-project.vercel.app/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "cgm_readings": [120, 122, 118, 121, 125, 128, 130, 135, 132, 129],
    "carbs": 30,
    "protein": 10,
    "fat": 5,
    "top_k": 10
  }'</pre>
            </div>
            
            <div class="links">
                <a href="/docs">📚 Swagger UI</a>
                <a href="/redoc">📖 ReDoc</a>
                <a href="/api/health">❤️ Health Check</a>
            </div>
            
            <div class="footer">
                Powered by FastAPI • Deployed on Vercel • v1.0.0
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    engine_status = {
        "loaded": _ENGINE is not None,
        "loading": _ENGINE_LOADING
    }
    
    return {
        "status": "healthy",
        "engine": engine_status,
        "message": "API is running" if _ENGINE is None else "Engine is ready",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict glucose risk from CGM readings
    
    Args:
        cgm_readings: List of 10 CGM readings (40-250 mg/dL)
        carbs: Carbohydrates in grams (optional, default 0)
        protein: Protein in grams (optional, default 0)
        fat: Fat in grams (optional, default 0)
        top_k: Number of food recommendations (optional, default 10)
    
    Returns:
        PredictionResponse with risk analysis and predictions
    """
    try:
        # Validate input
        if not request.cgm_readings or len(request.cgm_readings) == 0:
            raise HTTPException(
                status_code=400,
                detail="No CGM readings provided"
            )
        
        if len(request.cgm_readings) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected at least 10 CGM readings, got {len(request.cgm_readings)}"
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
# Vercel Handler (required for serverless)
# ============================================================
def handler(request, context):
    """Vercel serverless handler - wraps FastAPI app"""
    from mangum import Mangum
    return Mangum(app)(request, context)

# ============================================================
# Local Development
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
