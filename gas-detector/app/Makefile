# Compiler
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -pthread -I/usr/local/include

# Directories
APP_SRC_DIR      	  := src
SENSORS_INC_DIR  	  := ../sensors/include
INFERENCE_ENGINE_INC  := ../inference-engine/include
WINDOW_QUEUE_INC	  := ../window-queue/include
MLX_LIB_INC      	  := ../../mlx90640-library/headers
BUILD_DIR         	  := build

TF_SRC   := /home/isam/dev/MLX90640/tensorflow
TF_LIB   := /home/isam/dev/MLX90640/tflite_build

# Target
TARGET := gas_detector

# Sources (relative to app folder)
SRCS := src/main.cpp \
        ../sensors/src/MCP3008.cpp \
        ../sensors/src/MQSensor.cpp \
        ../sensors/src/MLX90640.cpp \
        ../inference-engine/src/inferenceEngine.cpp \
        ../window-queue/src/windowQueue.cpp \
        ../../mlx90640-library/functions/MLX90640_API.cpp \
        ../../mlx90640-library/functions/MLX90640_LINUX_I2C_Driver.cpp

# Objects (mirror folder structure in build/)
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Includes
INCLUDES := -I$(SENSORS_INC_DIR) \
            -I$(MLX_LIB_INC) \
			-I$(INFERENCE_ENGINE_INC) \
			-I$(WINDOW_QUEUE_INC) \
			-I${TF_SRC}

# Libraries - TensorFlow Lite with all dependencies
# Using whole-archive to ensure all symbols are linked
LDFLAGS := -Wl,--whole-archive \
           $(TF_LIB)/libtensorflow-lite.a \
           $(TF_LIB)/libxnnpack-delegate.a \
           $(TF_LIB)/pthreadpool/libpthreadpool.a \
           $(TF_LIB)/_deps/xnnpack-build/libXNNPACK.a \
           $(TF_LIB)/_deps/xnnpack-build/libmicrokernels-prod.a \
           $(TF_LIB)/_deps/xnnpack-build/libmicrokernels-all.a \
           $(TF_LIB)/kleidiai/libkleidiai.a \
           $(TF_LIB)/_deps/cpuinfo-build/libcpuinfo.a \
           $(TF_LIB)/_deps/farmhash-build/libfarmhash.a \
           $(TF_LIB)/_deps/fft2d-build/libfft2d_fftsg.a \
           $(TF_LIB)/_deps/fft2d-build/libfft2d_fftsg2d.a \
           $(TF_LIB)/_deps/fft2d-build/libfft2d_alloc.a \
           $(TF_LIB)/_deps/fft2d-build/libfft2d_shrtdct.a \
           $(TF_LIB)/_deps/flatbuffers-build/libflatbuffers.a \
           $(TF_LIB)/_deps/protobuf-build/libprotobuf.a \
           $(shell find $(TF_LIB)/_deps/ruy-build/ruy -maxdepth 1 -name '*.a' -type f) \
           $(shell find $(TF_LIB)/_deps/abseil-cpp-build -name '*.a' -type f) \
           -Wl,--no-whole-archive \
           -ldl -lpthread -lm

# Default target
all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built -> $(TARGET)"

# Compile rule (handles subdirs properly)
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Run
run: all
	./$(TARGET)
