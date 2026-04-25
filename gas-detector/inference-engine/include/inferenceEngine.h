// InferenceEngine
// 1. Load a .tflite model from disk
// 2. Load normalisation stats (mean/std for IR and gas)
// 3. Accept raw sensor data (ir_frames, gas)
// 4. Normalise the data using the stats
// 5. Feed it into the model
// 6. Read the output probabilities
// 7. Return a human-readable result

#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <vector>
#include <string>
#include <array>
#include <memory> 

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

struct InferenceResult {
    std::string          label;
    int                  classIdx;
    float                confidence;
    std::array<float, 4> probabilities;
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& modelPath);
    ~InferenceEngine() = default;   // unique_ptr handles cleanup

    // ir_frames : (20, 32, 24)
    // gas       : (20, 3)
    InferenceResult run(
        const std::vector<std::vector<std::vector<float>>>& irFrames,
        const std::vector<std::vector<float>>& gas
    );

private:
    void loadModel(const std::string& modelPath);
    void normalise(
        const std::vector<std::vector<std::vector<float>>>& irFrames,
        const std::vector<std::vector<float>>& gas,
        std::vector<float>& irNorm,    // output: flat (20*32*24)
        std::vector<float>& gasNorm    // output: flat (20*3)
    );

    // TFLite
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter>     interpreter_;

    static constexpr int   TIMESTEPS   = 20;
    static constexpr int   IR_H        = 32;
    static constexpr int   IR_W        = 24;
    static constexpr int   GAS_CH      = 3;
    static constexpr int   NUM_CLASSES = 4;

    static constexpr const char* CLASSES[4] = {
        "normal", "aerosol", "flame", "breath"
    };
};

#endif // INFERENCE_ENGINE_H