#include "inferenceEngine.h"
#include "normStats.h"
#include "tensorflow/lite/kernels/register.h"

#include <cstring>
#include <iostream>
#include <algorithm>
#include <stdexcept>


InferenceEngine::InferenceEngine(const std::string& modelPath) 
{
    loadModel(modelPath);
}

void InferenceEngine::loadModel(const std::string& modelPath) 
{
    // load model using tflite::FlatBufferModel::BuildFromFile
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    
    // check if model
    if (!model_) throw std::runtime_error("Model not loaded: " + modelPath);

    // Build the Interpreter
    // create a resolver
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // create a builder
    tflite::InterpreterBuilder builder(*model_, resolver);

    // check if builder(&interpreter)
    if (builder(&interpreter_) != kTfLiteOk || !interpreter_) {
        throw std::runtime_error("Failed to build interpreter");
    }

    // set interpreter num of threads
    interpreter_->SetNumThreads(2);

    // allocate tensors
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Tensor allocation failed");
    }

    std::cout << "[InferenceEngine] Loaded : " << modelPath << "\n";
}

void InferenceEngine::normalise(const std::vector<std::vector<std::vector<float>>>& irFrames,
               const std::vector<std::vector<float>>& gas,
               std::vector<float>& irNorm,  std::vector<float>& gasNorm) 
{
    // resize irNorm
    irNorm.resize(TIMESTEPS * IR_H * IR_W);

    // loop through each ir point (timesteps, height, width)
    for (int t = 0; t < TIMESTEPS; ++t) {
        for (int h = 0; h < IR_H; ++h) {
            for (int w = 0; w < IR_W; ++w) {
                // apply normalization
                // ir_norm  = (ir_frames - ir_mean) / ir_std
                int flatIdx = t * IR_H * IR_W + h * IR_W + w;
                int statIndex = h * IR_W + w;
                irNorm[flatIdx] = (irFrames[t][h][w] - IR_MEAN[statIndex]) / IR_STD[statIndex];
            }
        }
    }

    // resize gasNorm
    gasNorm.resize(TIMESTEPS * GAS_CH);
    
    // loop through each gas point (timesteps, channel)
    for (int t = 0; t < TIMESTEPS; ++t) {
        for (int ch = 0; ch < GAS_CH; ++ch) {
            // apply normalization
            // gas_norm = (gas - gas_mean) / gas_std
            gasNorm[t * GAS_CH + ch] = (gas[t][ch] - GAS_MEAN[ch]) / GAS_STD[ch];
        }
    }
}

InferenceResult InferenceEngine::run(
    const std::vector<std::vector<std::vector<float>>>& irFrames,
    const std::vector<std::vector<float>>& gas) 
{
    // Normalise 
    std::vector<float>  irNorm, gasNorm;
    normalise(irFrames, gas, irNorm, gasNorm);
    
    // copy irNorm and gasNorm to input tensor
    for (int idx : interpreter_->inputs()) {
        const std::string name = interpreter_->tensor(idx)->name;
        float* dest = interpreter_->typed_tensor<float>(idx);
        
        if (name.find("ir_frames") != std::string::npos) {
            std::copy(irNorm.begin(),  irNorm.end(),  dest);
        }
        else if (name.find("gas") != std::string::npos) {
            std::copy(gasNorm.begin(), gasNorm.end(), dest);
        } else {
            std::cerr << "[InferenceEngine] Unknown input: " << name << "\n";
        }
    }

    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Inference Invoke() failed");
    }

    float* output = interpreter_->typed_output_tensor<float>(0);

    InferenceResult result;

    std::copy(output, output + NUM_CLASSES, result.probabilities.begin());

    result.classIdx = std::max_element(result.probabilities.begin(), 
                                      result.probabilities.end())
                                      - result.probabilities.begin();
    result.label = CLASSES[result.classIdx];
    result.confidence = result.probabilities[result.classIdx];

    return result;
}
