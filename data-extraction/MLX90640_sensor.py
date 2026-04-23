import os
import sys
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to MLX90640 python wrapper
lib_path = os.path.join(BASE_DIR, "mlx90640-library/python/library")
build_path = os.path.join(
    BASE_DIR,
    "mlx90640-library/python/library/build/lib.linux-aarch64-cpython-313"
)

sys.path.insert(0, build_path)
sys.path.insert(0, lib_path)

import MLX90640 as mlx

# DONT CHANGE THIS BELOW! - only works when EXPECTED_FPS = 4 and #define FPS 16 in mlx90640-python.cpp
EXPECTED_FPS = 4

OUTPUT_NPZ = "mlx_ir_dataset.npz"
OUTPUT_CSV_AVG = "mlx_ir_avg.csv"
OUTPUT_CSV_FRAMES = "mlx_ir_frames.csv"

def getIrData(num_of_samples):
    mlx.setup(EXPECTED_FPS)
    
    frames = []

    # Capture data
    data = mlx.capture_frames(num_of_samples)   # returns a vector of 20 * 768 points
    frames = np.array(data).reshape(num_of_samples, 32, 24)
    
    # Drop first (bad) frame
    frames = frames[1:]
    
    # Cleanup
    mlx.cleanup()
    
    # AVERAGING
    print("\nAveraging frames...")
    stack = np.stack(frames, axis=0)   # (20, 32, 24)
    avg_frame = np.mean(stack, axis=0) # (32, 24)
    
    return avg_frame.astype(np.float32), frames
