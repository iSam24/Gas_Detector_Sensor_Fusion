import threading
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from data_extraction.MQgas_sensors import getGasData
from data_extraction.MLX90640_sensor import getIrData
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).parent.resolve()
MODEL_PATH      = str(_HERE / "model_float16.tflite")
NORM_STATS_PATH = str(_HERE / "norm_stats.npz")
EXPECTED_FPS    = 4
SAMPLE_DURATION = 5                        # seconds per window
TARGET_SAMPLES  = SAMPLE_DURATION * EXPECTED_FPS          # 20
TARGET_SAMPLES_IR = TARGET_SAMPLES + 1     # +1 because getIrData drops first frame
PROBABILITY_LIMIT = 0.60
CLASSES     = ["aerosol", "breath", "flame", "normal"]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Inputs : {[d['name'] for d in input_details]}")
print(f"Output shape: {output_details[0]['shape']}")

# ── Load normalisation stats ──────────────────────────────────────────────────
print("Loading normalisation stats...")
stats    = np.load(NORM_STATS_PATH)
ir_mean  = stats["ir_mean"].astype(np.float32)   # (1, 1, 32, 24)
ir_std   = stats["ir_std"].astype(np.float32)    # (1, 1, 32, 24)
gas_mean = stats["gas_mean"].astype(np.float32)  # (1, 1, 3)
gas_std  = stats["gas_std"].astype(np.float32)   # (1, 1, 3)

# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize(ir_frames, gas):
    """
    ir_frames : (20, 32, 24)
    gas       : (20, 3)
    """
    ir_norm  = (ir_frames - ir_mean) / ir_std
    gas_norm = (gas - gas_mean) / gas_std
    return ir_norm.astype(np.float32), gas_norm.astype(np.float32)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(ir_frames, gas):
    """
    ir_frames : (20, 32, 24) — already normalised
    gas       : (20, 3)      — already normalised
    """
    for d in input_details:
        if "ir_frames" in d["name"]:
            interpreter.set_tensor(d["index"], ir_frames)
        elif "gas" in d["name"]:
            interpreter.set_tensor(d["index"], gas)

    interpreter.invoke()

    probs     = interpreter.get_tensor(output_details[0]["index"])[0]  # (4,)
    pred_idx  = int(np.argmax(probs))
    return CLASSES[pred_idx], pred_idx, probs

# ── Capture window ────────────────────────────────────────────────────────────
ir_result  = {}
gas_result = {}

def capture_ir():
    # getIrData drops first frame internally, so request TARGET_SAMPLES + 1
    # returns: avg_frame (32,24), frames (TARGET_SAMPLES, 32, 24)
    avg, frames = getIrData(TARGET_SAMPLES_IR)
    ir_result["frames"] = frames.astype(np.float32)   # (20, 32, 24)

def capture_gas():
    # getGasData(duration_sec, fps) → (20, 3)
    data = getGasData(SAMPLE_DURATION, EXPECTED_FPS)
    gas_result["data"] = data.astype(np.float32)       # (20, 3)

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("\nStarting inference loop — Ctrl+C to stop\n")
    window_count = 0

    while True:
        window_count += 1
        print(f"[Window {window_count}] Capturing {SAMPLE_DURATION}s of sensor data...")

        # Capture IR and gas in parallel (they take ~5s each)
        t1 = threading.Thread(target=capture_ir)
        t2 = threading.Thread(target=capture_gas)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        ir_frames = ir_result["frames"]   # (20, 32, 24)
        gas_data  = gas_result["data"]    # (20, 3)
        
        # Validate shapes before inference
        if ir_frames.shape != (TARGET_SAMPLES, 32, 24):
            print(f"  WARNING: unexpected IR shape {ir_frames.shape}, skipping")
            continue
        if gas_data.shape != (TARGET_SAMPLES, 3):
            print(f"  WARNING: unexpected gas shape {gas_data.shape}, skipping")
            continue

        # Normalise
        ir_norm, gas_norm = normalize(ir_frames, gas_data)

        # Infer
        print(f"IR shape before inference:  {ir_norm.shape}")
        print(f"Gas shape before inference: {gas_norm.shape}")
        
        label, pred_idx, probs = run_inference(ir_norm, gas_norm)

        # Print results
        print(f"\n{'─'*45}")
        if probs[pred_idx] > PROBABILITY_LIMIT:
            print(f"  Prediction : {label.upper()}")
            print(f"  Confidence : {probs.max()*100:.1f}%")
        else:
            print(f"  Prediction : Uncertain")
            print(f"  Confidence : NA  ")

        print(f"  Breakdown  :")
        for i, cls in enumerate(CLASSES):
            bar = "█" * int(probs[i] * 20)
            print(f"    {cls:<10} {probs[i]*100:5.1f}%  {bar}")
        print(f"{'─'*45}\n")

if __name__ == "__main__":
    main()
    