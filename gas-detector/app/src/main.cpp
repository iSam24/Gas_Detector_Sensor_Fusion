#include <thread>
#include <vector>
#include <string>
#include <stdexcept>
#include <csignal>
#include <csignal>
#include <atomic>
#include <iostream>

#include "MLX90640.h"
#include "MQSensor.h"
#include "MCP3008.h"
#include "inferenceEngine.h"
#include "windowQueue.h"

// Config
static constexpr int   SAMPLE_DURATION_SEC = 5;
static constexpr int   EXPECTED_FPS        = 4;
static constexpr int   TARGET_SAMPLES      = SAMPLE_DURATION_SEC * EXPECTED_FPS;  // 20
static constexpr int   TARGET_SAMPLES_IR   = TARGET_SAMPLES + 1;                  // 21 — first dropped
static constexpr const char* MODEL_PATH    =   "/home/isam/dev/MLX90640/python-inference/model_float32.tflite";
static constexpr int    NUM_QUEUE_WINDOWS  = 2;

static std::atomic<bool> running{true};

void signalHandler(int sig) {
    std::cout << "Interrupt handle " << sig << "\n";
    running = false;
}

void captureThread(MLX90640& ir, MQSensor& gas, WindowQueue& queue) {
    while (running)
    {
        CaptureWindow window;

        std::exception_ptr irError, gasError;

        // Launch IR and gas capture in parallel
        std::thread irThread([&]() {
            try {
                window.irFrames = ir.getIrData(TARGET_SAMPLES_IR);
            } catch (...) {
                irError = std::current_exception();
            }
        });

        std::thread gasThread([&]() {
            try {
                window.gas = gas.getGasData(SAMPLE_DURATION_SEC, EXPECTED_FPS);
            } catch (...) {
                gasError = std::current_exception();
            }
        });

        irThread.join();
        gasThread.join();

        if (irError || gasError) {
            std::cerr << "[CaptureThread] Sensor error, skipping window\n";
            continue;
        }

        if ((int)window.irFrames.size() != TARGET_SAMPLES || 
            (int)window.gas.size() != TARGET_SAMPLES) {
            std::cerr << "[CaptureThread] Ir or Gas sample size incorrect,\n";
            continue; 
        }

        queue.push(std::move(window));
    }
    queue.stop();   // !running
}

void inferenceThread(InferenceEngine& engine, WindowQueue& queue) {
    while (true) 
    {
        // pop gas data
        auto window = queue.pop();

        if (!window) break; // queue stopped and empty

        InferenceResult result = engine.run(window->irFrames, window->gas);

        std::cout << "\n┌─────────────────────────────┐\n";
        std::cout << "│ Prediction: "
                  << result.label << " (" << result.confidence * 100.0f << "%)\n";
        std::cout << "├─────────────────────────────┤\n";
        const char* labels[] = {"normal", "aerosol", "flame", "breath"};
        for (int i = 0; i < 4; ++i) {
            int bar = static_cast<int>(result.probabilities[i] * 20);
            std::cout << "│ " << labels[i] << "\t"
                      << result.probabilities[i] * 100.0f << "%  "
                      << std::string(bar, '#') << "\n";
        }
        std::cout << "└─────────────────────────────┘\n";
    }
}

int main()
{
    std::signal(SIGINT, signalHandler);  // Ctrl + C handler

    MCP3008   adc("/dev/spidev0.0");
    MQSensor  gasSensor(adc);
    MLX90640  irSensor(EXPECTED_FPS);

    InferenceEngine engine(MODEL_PATH); 
    WindowQueue queue(NUM_QUEUE_WINDOWS);

    // start thread and pass args by ref
    std::thread capture(captureThread, std::ref(irSensor), std::ref(gasSensor), std::ref(queue));
    std::thread inference(inferenceThread, std::ref(engine), std::ref(queue));

    capture.join();
    inference.join();

    return 0;
}