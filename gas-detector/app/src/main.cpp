#include <thread>
#include <vector>
#include <string>
#include <stdexcept>
#include <csignal>
#include <csignal>
#include <atomic>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "MLX90640.h"
#include "MQSensor.h"
#include "MCP3008.h"
#include "inferenceEngine.h"
#include "windowQueue.h"

using json = nlohmann::json;

// Config
static constexpr int   SAMPLE_DURATION_SEC = 5;
static constexpr int   EXPECTED_FPS        = 4;
static constexpr int   TARGET_SAMPLES      = SAMPLE_DURATION_SEC * EXPECTED_FPS;  // 20
static constexpr int   TARGET_SAMPLES_IR   = TARGET_SAMPLES + 1;                  // 21 — first dropped
static constexpr const char* MODEL_PATH    =   "/home/isam/dev/MLX90640/python-inference/model_float32v2.tflite";
static constexpr int    NUM_QUEUE_WINDOWS  = 2;
static bool             DEBUG              = false;
static constexpr const char* FRONTEND_URL  = "http://localhost:8000/api/data";
static bool             ENABLE_FRONTEND    = true;

static std::atomic<int> capture_count{0};
static std::atomic<bool> running{true};

void signalHandler(int sig) {
    std::cout << "Interrupt handle " << sig << "\n";
    running = false;
}

// Callback for CURL to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Send prediction data to frontend via HTTP POST
void sendToFrontend(const std::vector<std::vector<std::vector<float>>>& irFrames,
                     const std::vector<std::vector<float>>& gasData,
                     const InferenceResult& result)
{
    if (!ENABLE_FRONTEND) return;

    try {
        // Flatten IR frames (20 frames x 768 values each)
        json irDataJson = json::array();
        for (const auto& frame : irFrames) {
            std::vector<float> flatFrame;
            for (const auto& row : frame) {
                for (float val : row) {
                    flatFrame.push_back(val);
                }
            }
            irDataJson.push_back(flatFrame);
        }

        // Convert gas data
        json gasDataJson = gasData;

        // Create payload
        json payload = {
            {"ir_data", irDataJson},
            {"gas_data", gasDataJson},
            {"prediction", result.label},
            {"confidence", result.confidence},
            {"probabilities", result.probabilities},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
        };

        std::string jsonStr = payload.dump();

        // Send via CURL
        CURL* curl = curl_easy_init();
        if (!curl) {
            std::cerr << "[Frontend] Failed to initialize CURL\n";
            return;
        }

        std::string readBuffer;
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, FRONTEND_URL);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 2L);

        CURLcode res = curl_easy_perform(curl);
        
        if (res != CURLE_OK) {
            std::cerr << "[Frontend] Failed to send data: " << curl_easy_strerror(res) << "\n";
        } else {
            if (DEBUG) {
                std::cout << "[Frontend] Data sent successfully (" << jsonStr.length() << " bytes)\n";
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    } catch (const std::exception& e) {
        std::cerr << "[Frontend] Error: " << e.what() << "\n";
    }
}

// Write IR frames (20, 32, 24) as (20, 768) flat CSV — matches debug_ir_frames.csv
void writeIrCsv(const std::string& path,
                const std::vector<std::vector<std::vector<float>>>& frames)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    f << std::fixed;
    f.precision(2);

    for (const auto& frame : frames) {           // 20 frames
        bool firstVal = true;
        for (const auto& row : frame) {          // 32 rows
            for (float v : row) {                // 24 cols
                if (!firstVal) f << ",";
                f << v;
                firstVal = false;
            }
        }
        f << "\n";
    }

    std::cout << "Saved IR CSV  → " << path
              << "  (" << frames.size() << " x 768)\n";
}

// Write gas data (20, 3) CSV — matches debug_gas.csv
void writeGasCsv(const std::string& path,
                 const std::vector<std::vector<float>>& gas)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    f << std::fixed;
    f.precision(2);

    for (const auto& sample : gas) {            // 20 samples
        for (size_t i = 0; i < sample.size(); ++i) {
            if (i > 0) f << ",";
            f << sample[i];
        }
        f << "\n";
    }

    std::cout << "Saved gas CSV → " << path
              << "  (" << gas.size() << " x " << (gas.empty() ? 0 : gas[0].size()) << ")\n";
}

void captureThread(MLX90640& ir, MQSensor& gas, WindowQueue& queue) {

    if (DEBUG) {
        std::filesystem::create_directories("debug_csv");
    }

    while (running)
    {
        capture_count++;
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

        if (DEBUG) {
            try {
                std::string irCsvPath = "debug_csv/debug_ir_cpp" + std::to_string(capture_count) + ".csv";
                std::string gasCsvPath = "debug_csv/debug_gas_cpp" + std::to_string(capture_count) + ".csv";
                writeIrCsv(irCsvPath,  window.irFrames);
                writeGasCsv(gasCsvPath, window.gas);
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to write CSV: " << e.what() << "\n";
                continue;
            } 
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

        // Send to frontend
        sendToFrontend(window->irFrames, window->gas, result);

        std::cout << "\n┌─────────────────────────────┐\n";
        std::cout << "│ Prediction: "
                  << result.label << " (" << result.confidence * 100.0f << "%)\n";
        std::cout << "├─────────────────────────────┤\n";
        const char* labels[] = {"aerosol", "flame", "normal"};
        for (int i = 0; i < 3; ++i) {
            std::cout << "│ " << labels[i] << "\t"
                      << result.probabilities[i] * 100.0f << "%  " << "\n";
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
