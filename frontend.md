# Frontend Implementation

## Data Flow Architecture

```
Sensors (I2C/SPI)
         ↓
    C++ Application
    ├─ Capture Thread (IR + Gas)
    ├─ Inference Thread (TFLite)
       └─ sendToFrontend() [HTTP POST]
         ↓
    Python FastAPI Server :8000
    ├─ Receive POST endpoint
    ├─ Parse JSON data
    └─ Broadcast via WebSocket
         ↓
    Browser Dashboard
    ├─ Connect via WebSocket
    ├─ Receive IR frames + Gas data
    └─ Display in real-time
```

## Setup

#### Install curl development libraries (for C++ HTTP client)
sudo apt-get install -y libcurl4-openssl-dev

#### Install nlohmann/json (C++ JSON library)
sudo apt-get install -y nlohmann-json3-dev

## Run

```bash
Terminal 1
cd gas-detector/app

./gas_detector

Terminal 2
cd python-frontend/

source venv/bin/activate

python fastapi_server.py
```

## Access frontend

```
http://localhost:8000
```

or from another machine on the network:

```
http://<raspberry-pi-ip>:8000
```
