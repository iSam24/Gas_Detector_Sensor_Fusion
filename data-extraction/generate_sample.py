import threading
import uuid
import os
import numpy as np

from MQgas_sensors import getGasData
from MLX90640_sensor import getIrData

SAMPLE_DURATION_SEC = 5

# DONT CHANGE THIS BELOW
# only works when EXPECTED_FPS = 4 and #define FPS 16 in mlx90640-python.cpp
EXPECTED_FPS = 4
TARGET_SAMPLES = SAMPLE_DURATION_SEC * EXPECTED_FPS
TARGET_SAMPLES_IR = TARGET_SAMPLES + 1 # Get 1 more sample as the first sample is dropped as it contains 0's

DATASET_DIR = "dataset"
LABEL = "test"   # <-- change per class

ir_result = {}
gas_result = {}

def capture_ir():
    ir_avg, ir_frames = getIrData(TARGET_SAMPLES_IR)
    ir_result["avg"] = ir_avg
    ir_result["frames"] = ir_frames

def capture_gas():
    gas_data = getGasData(SAMPLE_DURATION_SEC, EXPECTED_FPS)
    gas_result["data"] = gas_data
    

def main():
    # Create label directory
    label_dir = os.path.join(DATASET_DIR, LABEL)
    os.makedirs(label_dir, exist_ok=True)

    print("Starting synchronized capture...")

    t1 = threading.Thread(target=capture_ir)
    t2 = threading.Thread(target=capture_gas)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Retrieve results
    ir_avg = ir_result["avg"].astype(np.float32)
    ir_frames = ir_result["frames"].astype(np.float32)
    gas_data = gas_result["data"].astype(np.float32)

    # Generate filename
    sample_id = str(uuid.uuid4())[:8]
    filename = os.path.join(label_dir, f"sample_{sample_id}.npz")

    # Save combined sample
    # {
    # "ir": (32, 24),
    # "ir_frames": (20, 32, 24),
    # "gas": (20, 3)
    # }
    np.savez_compressed(
        filename,
        ir=ir_avg,           # (32,24)
        ir_frames=ir_frames, # (20,32,24)
        gas=gas_data         # (20,3)
    )
    
    # Csv for debugging
    np.savetxt("debug_ir_avg.csv", ir_avg, delimiter=",", fmt="%.2f")
    np.savetxt("debug_gas.csv", gas_data, delimiter=",", fmt="%.2f")
    
    # Flatten ir_frames into (20,768)
    frames_flat = ir_frames.reshape(TARGET_SAMPLES, -1)
    np.savetxt("debug_ir_frames.csv", frames_flat, delimiter=",", fmt="%.2f")

    print(f"\nSaved sample → {filename}")
    print(f"IR shape: {ir_avg.shape}")
    print(f"Gas shape: {gas_data.shape}")

if __name__ == "__main__":
    main()
