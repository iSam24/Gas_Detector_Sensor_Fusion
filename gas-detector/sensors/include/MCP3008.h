#ifndef MCP3008_H
#define MCP3008_H

class MCP3008 {
public:
    MCP3008(const std::string&  = "/dev/spidev0.0", uint32_t speed = 1350000);
    ~MCP3008();
    
    int readChannel(int channel);

private:
    int spi_fd;
    int speed = 135000;
};

#endif // MCP3008_H
