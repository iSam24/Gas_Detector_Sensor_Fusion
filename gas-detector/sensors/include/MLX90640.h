#ifndef MLX90640_H
#define MLX90640_H

#include <vector>
#include "MLX90640_API.h"

#define MLX_I2C_ADDR 0x33

// Despite the framerate being ostensibly FPS hz
// The frame is often not ready in time
// This offset is added to the frame time
// to account for this.
#define OFFSET_MICROS 850

class MLX90640 {
public:
    MLX90640(int fps);
 
    // Returns (numSamples-1, 32, 24) — first frame dropped
    std::vector<std::vector<std::vector<float>>> getIrData(int numSamples);

private:
    int setup();
    void captureIntoBuffer(int numFrames, std::vector<float>& buffer);

    int     fps;
    int     baudrate;
    float   emissivity = 1.0f;
    float   eTa;

    paramsMLX90640 mlx90640;
    static uint16_t eeMLX90640[832];
    uint16_t frame[834];
    static float mlx90640To[768];
};

#endif // MLX90640_H
