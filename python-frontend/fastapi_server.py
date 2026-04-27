"""
FastAPI WebSocket Relay Server

Receives sensor data from C++ backend via HTTP POST
Broadcasts to connected WebSocket clients (browsers)

Architecture:
  C++ (Backend) ---HTTP POST---> FastAPI Server ---WebSocket---> Browser
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Set
from pathlib import Path
import base64
import numpy as np

# Configuration
HOST = "0.0.0.0"
PORT = 8000
CPP_POST_PORT = 8765

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Gas Detector Frontend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.latest_data = None
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        
        # Send latest data to newly connected client
        if self.latest_data:
            try:
                await websocket.send_json(self.latest_data)
            except Exception as e:
                logger.error(f"Failed to send initial data: {e}")

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """Broadcast data to all connected clients"""
        async with self.lock:
            self.latest_data = data
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)

# Global connection manager
manager = ConnectionManager()

# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.get("/")
async def get_frontend():
    """Serve the main frontend HTML"""
    frontend_path = Path(__file__).parent / "templates" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path, media_type="text/html")
    return HTMLResponse("""
        <h1>Gas Detector Frontend Loading...</h1>
        <p>Frontend HTML not found. Ensure index.html is in templates/ directory.</p>
    """)

@app.post("/api/data")
async def receive_sensor_data(data: dict):
    """
    Receive sensor data from C++ backend
    
    Expected JSON format:
    {
        "ir_data": [[...], ...],  # 20x768 flattened frames
        "gas_data": [[...], ...],  # 20x3 gas readings
        "prediction": "normal",
        "confidence": 0.95,
        "probabilities": [0.95, 0.03, 0.02],
        "timestamp": 1234567890
    }
    """
    try:
        # Validate required fields
        required_fields = ["ir_data", "gas_data", "prediction", "confidence", "probabilities"]
        if not all(field in data for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Add server-side timestamp
        data["server_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Received prediction: {data['prediction']} ({data['confidence']*100:.1f}%)")
        
        # Broadcast to all connected WebSocket clients
        await manager.broadcast(data)
        
        return {"status": "success", "message": "Data received and broadcasted"}
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get server status and connected clients"""
    return {
        "server": "running",
        "connected_clients": len(manager.active_connections),
        "latest_data": "available" if manager.latest_data else "none"
    }

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for browser connections
    
    Receives: Real-time sensor data and inference results
    Broadcasts: To all connected clients
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive any keepalive messages
            data = await websocket.receive_text()
            if data.lower() == "ping":
                await websocket.send_text("pong")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)

# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/api/config")
async def get_config():
    """Get frontend configuration"""
    return {
        "ir_resolution": [32, 24],
        "num_gas_sensors": 3,
        "gas_labels": ["MQ2", "MQ135", "MQ7"],
        "classes": ["aerosol", "flame", "normal"],
        "update_interval_ms": 5000,
        "graph_max_points": 100
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Gas Detector Frontend Server on {HOST}:{PORT}")
    logger.info(f"WebSocket: ws://{HOST}:{PORT}/ws")
    logger.info(f"Data Endpoint: http://{HOST}:{PORT}/api/data")
    logger.info(f"Open browser to: http://{HOST}:{PORT}")
    
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
