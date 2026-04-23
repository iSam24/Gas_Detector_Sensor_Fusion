# Data Collection

The generate_sample.py script is used to generate a sample and store sample data in a numpy .npz file. The folder should be configured in the generate_sample.py file as this will be the class name for the model. 

## Classes

### Normal
Source: NA
Baseline null reference with no gas at standard room conditions.

### Flame
Source: lighter. Long duration (~3s)
Gas: CO presence (MQ7), combustion gases (MQ2 rise)
IR: Strong localised hot spot with flickering thermal pattern

### Aerosol
Source: Deodorant Spray. Long duration spray (~ 3s)
Gas: strong VOC spike (MQ135 especially)
IR: cold plume (evaporative cooling)

### Breath
Source: Breathe onto sensors
Gas: CO₂ proxy (MQ135 rises), humidity spike
IR: warm (~34–36°C), diffuse plume

## Control variables for data collection

### Distance from sensors

All sources shall be 30cm from the sensors during data collection. There shall be no heat sources in the IR cameras view other than the source whilst collecting data.

### Environment

All data will be collected in a room with still air at around 19 degrees celcius.
