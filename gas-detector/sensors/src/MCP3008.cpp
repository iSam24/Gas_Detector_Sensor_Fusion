#include "MCP3008.h"

#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <cstdint>

MCP3008::MCP3008(const std::string& device, uint32_t speed) : speed(speed) {
    spi_fd = open(device.c_str(), O_RDWR);  // open file decriptor
    if (spi_fd < 0) {
        throw std::runtime_error("Failed to open SPI device");
    }

    // Init spi
    uint8_t mode = SPI_MODE_0;
    uint8_t bits = 8;

    if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) == -1) {
        close(spi_fd);
        throw std::runtime_error("Failed to set SPI mode");
    }

    if (ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) == -1) {
        close(spi_fd);
        throw std::runtime_error("Failed to set SPI bit per word");
    }

    if (ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) == -1) {
        close(spi_fd);
        throw std::runtime_error("Failed to set SPI max speed");
    }
}

MCP3008::~MCP3008() {
    if (spi_fd >= 0) {
        close(spi_fd);
    }
}

int MCP3008::readChannel(int channel) {
    // MCP3008 has 8 channels: 0–7
    if (channel < 0 || channel > 7) {
        throw std::out_of_range("MCP3008 channel must be 0-7");
    }

    // MCP3008 expects a 3-byte SPI transaction:
    //
    // Byte 1: Start bit (always 1)
    // Byte 2: Configuration:
    //   - Single-ended mode (1)
    //   - Channel selection (3 bits)
    // Byte 3: Don't care (just clock pulses to receive data)
    //
    // Format (from datasheet):
    // [00000001] [SGL/DIF | D2 | D1 | D0 | xxxx] [xxxxxxxx]
    //
    // (8 + channel) << 4 sets:
    // - SGL/DIF = 1 (single-ended)
    // - D2 D1 D0 = channel bits
    // example for channel 3:
    // 8 + 3 = 11 = 0b1011
    //                ||||
    //                ||||- D0 channel bit
    //                |||-- D1 channel bit
    //                ||---- D2 channel bit
    //                |------ single-ended mode 
    // << 4 to convert to 8 bit as MCP3008 expects the 4 MSB to contain the data
    uint8_t tx[3] = {1, static_cast<uint8_t>((8 + channel) << 4), 0};
    
    // Buffer to receive data from SPI
    uint8_t rx[3] = {0};

    // SPI transaction structure
    struct spi_ioc_transfer tr{};
    tr.tx_buf = reinterpret_cast<unsigned long>(tx); // transmit buffer
    tr.rx_buf = reinterpret_cast<unsigned long>(rx); // receive buffer
    tr.len = 3;                                      // 3 bytes transfer
    tr.speed_hz = speed;                             // SPI clock speed
    tr.bits_per_word = 8;

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr) < 1) {
        throw std::runtime_error("Failed to send MCP3008 spi ioctl message");
    }

    // MCP3008 returns 10-bit result across rx[1] and rx[2]:
    //
    // rx[0] = rubbish
    // rx[1] = xxxx xxBB   (lower 2 bits are data bits 9 and 8)
    // rx[2] = BBBB BBBB   (lower 8 bits are data bits 7–0)
    //
    // Combine into 10-bit integer:
    int value = ((rx[1] & 3) << 8) | rx[2];

    return value; // range: 0–1023
}
