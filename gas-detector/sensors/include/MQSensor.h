#ifndef MQ_SENSOR_H
#define MQ_SENSOR_H

#include "MCP3008.h"

#include <vector>

namespace mq {
    enum class SensorType {
        MQ2 = 0,
        MQ7 = 1,
        MQ135 = 2
    };

    constexpr int NUM_SENSORS = 3;
}

class MQSensor {
public: 
    MQSensor(MCP3008& spiAdc); // Constructor

    // Returns [num_samples][num_sensors]
    std::vector<std::vector<float>> getGasData(float sampleTime, float sampleFrequency);

private:
    float toVoltage(int raw);
    float getRatio(float Vout, float RL);
    float getPPM(int ch, int raw);
    inline int getChannel(mq::SensorType sensor);

    MCP3008& adc;
    static constexpr float VREF = 3.3;
    static const int ADC_MAX = 1023;
    static constexpr int MQ2_RL = 5000;    // load resistance (ohms)
    static constexpr int MQ7_RL = 10000;   // load resistance (ohms)
    static constexpr int MQ135_RL = 10000; // load resistance (ohms)

    // Resistance at clean air baseline
    static constexpr float R0_MQ2 = 13184.66;
    static constexpr float R0_MQ7 = 419.98;
    static constexpr float R0_MQ135 = 112499.16;

    // log-log curve fitting constants (CHECK THESE)
    static constexpr float A_MQ7 = 99.042;
    static constexpr float B_MQ7 = -1.518;
    static constexpr float A_MQ135 = 116.6020682;
    static constexpr float B_MQ135 = -2.769034857;
    static constexpr float A_MQ2 = 574.25;
    static constexpr float B_MQ2 = -2.222;

    // Clean air ratio (R0_MQ7 = rs_clean_air / CLEAN_AIR_RATIO) clean air ratio is found in the datasheet
    static constexpr float MQ2_CLEAN_AIR_RATIO = 9.83;
    static constexpr float MQ7_CLEAN_AIR_RATIO = 27.5;
    static constexpr float MQ135_CLEAN_AIR_RATIO = 3.6;

    //  ppm explanation :
    //  The graph in the datasheet is represented with the function
    //  f(x) = a * (x ^ b).
    //  	where
    //  		f(x) = ppm
    //  		x = Rs/R0
    //  	The values were mapped with this function to determine the coefficients a and b.
    // from https://github.com/swatish17/MQ7-Library/blob/master/MQ7.h
    static constexpr float MQ7_coeff_A = 19.32;
    static constexpr float MQ7_coeff_B = -0.64;

};


#endif // MQ_SENSOR_H