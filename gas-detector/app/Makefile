# Compiler
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -pthread

# Directories
APP_SRC_DIR      := src
SENSORS_SRC_DIR  := ../sensors/src
SENSORS_INC_DIR  := ../sensors/include
MLX_LIB_INC      := ../../mlx90640-library/headers
APP_INC_DIR      := include
BUILD_DIR        := build

# Target
TARGET := gas_detector

# Sources (relative to app folder)
SRCS := src/main.cpp \
        ../sensors/src/MCP3008.cpp \
        ../sensors/src/MQSensor.cpp \
        ../sensors/src/MLX90640.cpp \
        ../../mlx90640-library/functions/MLX90640_API.cpp \
        ../../mlx90640-library/functions/MLX90640_LINUX_I2C_Driver.cpp

# Objects (mirror folder structure in build/)
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Includes
INCLUDES := -I$(APP_INC_DIR) \
            -I$(SENSORS_INC_DIR) \
            -I$(MLX_LIB_INC)

# Libraries (SPI + math + pthread)
LDFLAGS := -lpthread

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
