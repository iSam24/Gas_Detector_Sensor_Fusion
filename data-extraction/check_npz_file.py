import numpy as np

FILE_PATH = "dataset/normal/sample_faaa749a.npz"

def main():
    print(f"Loading file: {FILE_PATH}\n")

    data = np.load(FILE_PATH)

    print("Keys in file:")
    print(list(data.keys()))
    print()

    # ---- IR Average ----
    if "ir" in data:
        ir = data["ir"]
        print("IR Average:")
        print(f"Shape: {ir.shape}")
        print(ir)
        print()

    # ---- IR Frames ----
    if "ir_frames" in data:
        ir_frames = data["ir_frames"]
        print("IR Frames:")
        print(f"Shape: {ir_frames.shape}")
        print("First frame:")
        print(ir_frames[0])
        print()

    # ---- Gas ----
    if "gas" in data:
        gas = data["gas"]
        print("Gas Data:")
        print(f"Shape: {gas.shape}")
        print(gas)
        print()

    # ---- Sanity Checks ----
    print("Sanity checks:")
    
    if "ir" in data:
        print(f"IR min/max: {ir.min():.2f} / {ir.max():.2f}")
    
    if "gas" in data:
        print(f"Gas min/max: {gas.min():.2f} / {gas.max():.2f}")
    
    if "ir_frames" in data:
        zero_count = np.sum(ir_frames == 0)
        print(f"IR zero values: {zero_count}")

if __name__ == "__main__":
    main()