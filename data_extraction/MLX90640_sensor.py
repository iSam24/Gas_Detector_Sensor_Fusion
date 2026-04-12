# import sys
# import time
# import numpy as np

# # Path to MLX90640 python wrapper (adjust if needed)
# sys.path.insert(0, "./build/lib.linux-armv7l-2.7")

# import MLX90640 as mlx

# # CONFIG
# SAMPLE_DURATION_SEC = 5          # total capture window

# # DONT CHANGE THIS BELOW! - only works when EXPECTED_FPS = 4 and #define FPS 16 in mlx90640-python.cpp
# EXPECTED_FPS = 4
# TARGET_SAMPLES = SAMPLE_DURATION_SEC * EXPECTED_FPS

# OUTPUT_NPZ = "mlx_ir_dataset.npz"
# OUTPUT_CSV_AVG = "mlx_ir_avg.csv"
# OUTPUT_CSV_FRAMES = "mlx_ir_frames.csv"

# # Set refresh rate
# mlx.setup(EXPECTED_FPS)

# # STORAGE
# frames = []

# # CAPTURE DATA
# data = mlx.capture_frames(TARGET_SAMPLES)   # returns a vector of 20 * 768 points
# frames = np.array(data).reshape(TARGET_SAMPLES, 32, 24)

# # CLEANUP
# mlx.cleanup()

# # AVERAGING
# print("\nAveraging frames...")

# stack = np.stack(frames, axis=0)   # (20, 32, 24)
# avg_frame = np.mean(stack, axis=0) # (32, 24)

# # SAVE DATASET
# np.savez_compressed(
#     OUTPUT_NPZ,
#     ir=avg_frame.astype(np.float32),
#     num_frames=len(frames),
#     duration=SAMPLE_DURATION_SEC
# )

# # SAVE CSV (average frame)
# # Save as 32 rows x 24 cols
# np.savetxt(OUTPUT_CSV_AVG, avg_frame, delimiter=",", fmt="%.2f")

# # SAVE CSV (all frames flattened)
# # shape: (num_frames, 768)
# frames_flat = frames.reshape(TARGET_SAMPLES, -1)
# np.savetxt(OUTPUT_CSV_FRAMES, frames_flat, delimiter=",", fmt="%.2f")

# print(f"Shape: {avg_frame.shape}")
# print(f"Frames used: {len(frames)}")
# print(f"- NPZ: {OUTPUT_NPZ}")
# print(f"- CSV (avg): {OUTPUT_CSV_AVG}  → shape {avg_frame.shape}")
# print(f"- CSV (frames): {OUTPUT_CSV_FRAMES} → shape {frames_flat.shape}")





import sys
import time
import numpy as np

# Path to MLX90640 python wrapper (adjust if needed)
sys.path.insert(0, "./build/lib.linux-armv7l-2.7")

import MLX90640 as mlx

# CONFIG
SAMPLE_DURATION_SEC = 5

# DONT CHANGE THIS BELOW! - only works when EXPECTED_FPS = 4 and #define FPS 16 in mlx90640-python.cpp
EXPECTED_FPS = 4
TARGET_SAMPLES = SAMPLE_DURATION_SEC * EXPECTED_FPS

OUTPUT_NPZ = "mlx_ir_dataset.npz"
OUTPUT_CSV_AVG = "mlx_ir_avg.csv"
OUTPUT_CSV_FRAMES = "mlx_ir_frames.csv"

def getIrData(num_of_samples):
    mlx.setup(EXPECTED_FPS)
    
    frames = []

    # CAPTURE DATA
    data = mlx.capture_frames(TARGET_SAMPLES)   # returns a vector of 20 * 768 points
    frames = np.array(data).reshape(TARGET_SAMPLES, 32, 24)
    
    # CLEANUP
    mlx.cleanup()
    
    # AVERAGING
    print("\nAveraging frames...")
    stack = np.stack(frames, axis=0)   # (20, 32, 24)
    avg_frame = np.mean(stack, axis=0) # (32, 24)
    
    return avg_frame.astype(np.float32), frames
