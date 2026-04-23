#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <stdexcept>

#include "MLX90640.h"
#include "MQSensor.h"
#include "MCP3008.h"

// ── Config (mirrors generate_sample.py) ──────────────────────────────────────
static constexpr int   SAMPLE_DURATION_SEC = 5;
static constexpr int   EXPECTED_FPS        = 4;
static constexpr int   TARGET_SAMPLES      = SAMPLE_DURATION_SEC * EXPECTED_FPS;  // 20
static constexpr int   TARGET_SAMPLES_IR   = TARGET_SAMPLES + 1;                  // 21 — first dropped

// ── CSV helpers ───────────────────────────────────────────────────────────────

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

// ── Shared capture results ────────────────────────────────────────────────────
struct CaptureResult {
    std::vector<std::vector<std::vector<float>>> irFrames;  // (20, 32, 24)
    std::vector<std::vector<float>>              gasData;   // (20, 3)
    std::exception_ptr                           irError;
    std::exception_ptr                           gasError;
};

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "Initialising sensors...\n";

    // SPI ADC for gas sensors — adjust bus/device to match your wiring
    MCP3008 adc("/dev/spidev0.0");
    MQSensor  gassensor(adc);
    MLX90640  irSensor(EXPECTED_FPS);

    CaptureResult result;

    std::cout << "Starting synchronised capture ("
              << TARGET_SAMPLES << " samples @ "
              << EXPECTED_FPS << " Hz)...\n";

    // Launch IR and gas capture in parallel — mirrors t1/t2 in Python
    std::thread irThread([&]() {
        try {
            result.irFrames = irSensor.getIrData(TARGET_SAMPLES_IR);
        } catch (...) {
            result.irError = std::current_exception();
        }
    });

    std::thread gasThread([&]() {
        try {
            result.gasData = gassensor.getGasData(SAMPLE_DURATION_SEC, EXPECTED_FPS);
        } catch (...) {
            result.gasError = std::current_exception();
        }
    });

    irThread.join();
    gasThread.join();

    // Re-throw any capture errors
    if (result.irError) {
        try { std::rethrow_exception(result.irError); }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] IR capture failed: " << e.what() << "\n";
            return 1;
        }
    }
    if (result.gasError) {
        try { std::rethrow_exception(result.gasError); }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Gas capture failed: " << e.what() << "\n";
            return 1;
        }
    }

    // Validate shapes
    if ((int)result.irFrames.size() != TARGET_SAMPLES) {
        std::cerr << "[ERROR] Expected " << TARGET_SAMPLES
                  << " IR frames, got " << result.irFrames.size() << "\n";
        return 1;
    }
    if ((int)result.gasData.size() != TARGET_SAMPLES) {
        std::cerr << "[ERROR] Expected " << TARGET_SAMPLES
                  << " gas samples, got " << result.gasData.size() << "\n";
        return 1;
    }

    // Write CSVs
    try {
        writeIrCsv ("debug_ir_cpp.csv",  result.irFrames);
        writeGasCsv("debug_gas_cpp.csv", result.gasData);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to write CSV: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nDone.";
    
    return 0;
}