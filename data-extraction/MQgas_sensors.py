import spidev
import time
import numpy as np

spi = spidev.SpiDev()
spi.open(0, 0)          # bus 0, CE0
spi.max_speed_hz = 1350000

VREF = 3.3
ADC_MAX = 1023
MQ2_RL = 5000    # load resistance (ohms)
MQ7_RL = 10000   # load resistance (ohms)
MQ135_RL = 10000 # load resistance (ohms)

# Resistance at clean air baseline
R0_MQ2 = 13184.66
R0_MQ7 = 419.98
R0_MQ135 = 112499.16

# log-log curve fitting constants (CHECK THESE)
A_MQ7 = 99.042
B_MQ7 = -1.518
A_MQ135 = 116.6020682
B_MQ135 = -2.769034857
A_MQ2 = 574.25
B_MQ2 = -2.222

# Clean air ratio (R0_MQ7 = rs_clean_air / CLEAN_AIR_RATIO) clean air ratio is found in the datasheet
MQ2_CLEAN_AIR_RATIO = 9.83
MQ7_CLEAN_AIR_RATIO = 27.5
MQ135_CLEAN_AIR_RATIO = 3.6

# ppm explanation :
# 	The graph in the datasheet is represented with the function
# 	f(x) = a * (x ^ b).
# 	where
# 		f(x) = ppm
# 		x = Rs/R0
# 	The values were mapped with this function to determine the coefficients a and b.

# from https://github.com/swatish17/MQ7-Library/blob/master/MQ7.h
MQ7_coefficient_A = 19.32
MQ7_coefficient_B = -0.64

CHANNELS = {
    "MQ-2  (combustible)": 0,
    "MQ-7  (CO)         ": 1,
    "MQ-135 (air quality)": 2,
}

def read_channel(ch):
    """Read a value (0-1023) from MCP3008 channel 0-7"""
    if ch < 0 or ch > 7:
        return -1
    r = spi.xfer2([1, (8 + ch) << 4, 0])
    return ((r[1] & 3) << 8) + r[2]

def to_voltage(raw):
    """Convert raw ADC value to voltage"""
    return (raw / 1023.0) * VREF

def getRatio(V_out, RL):
    if V_out == 0:
        return None
    return RL * (VREF - V_out) / V_out

def getPPM(ch, raw):
    if raw >= 1023:
        raw = 1023

    volts = to_voltage(raw)

    if ch == 0:
        rs = getRatio(volts, MQ2_RL)
        return A_MQ2 * ((rs / R0_MQ2) ** B_MQ2)

    elif ch == 1:
        rs = getRatio(volts, MQ7_RL)
        ratio = rs / R0_MQ7
        # prevent zero or negative ratio
        ratio = max(ratio, 1e-5)
        return MQ7_coefficient_A * ((ratio) ** MQ7_coefficient_B)

    elif ch == 2:
        rs = getRatio(volts, MQ135_RL)
        return A_MQ135 * ((rs / R0_MQ135) ** B_MQ135)

    return -1


def getGasData(total_sample_time=5, sample_frequency=4):
    """
    Samples gas sensors over time and returns:
    - raw time series in ppm format

    Returns:
        data: shape (num_samples, num_sensors) (20,3)
    """
    num_samples = total_sample_time * sample_frequency
    sample_interval = 1.0 / sample_frequency

    sensor_names = list(CHANNELS.keys())
    num_sensors = len(sensor_names)
    
    # Storage: rows = time, cols = sensors
    data = np.zeros((num_samples, num_sensors), dtype=np.float32)

    print(f"Sampling gas sensors: {num_samples} samples @ {sample_frequency}Hz")

    for i in range(num_samples):
        start = time.time()
        
        for j, (name, ch) in enumerate(CHANNELS.items()):
            raw = read_channel(ch)
            ppm = getPPM(ch, raw)
            
            if ppm is None:
                ppm = 0
                
            data[i, j] = ppm
            
        elapsed = time.time() - start
        if sample_interval > elapsed:
            time.sleep(sample_interval - elapsed)
            
    return data        
    