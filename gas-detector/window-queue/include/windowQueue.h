#ifndef WINDOW_QUEUE_H
#define WINDOW_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <vector>

struct CaptureWindow {
    std::vector<std::vector<std::vector<float>>> irFrames;  // (20, 32, 24)
    std::vector<std::vector<float>>              gas;       // (20, 3)
};

class WindowQueue {
public:
    WindowQueue(int maxSize = 2) : maxSize_(maxSize) {};
    ~WindowQueue() = default;

    // Called by capture thread
    void push(CaptureWindow window);

    // called by inference thread
    std::optional<CaptureWindow> pop();

    // locks the queue
    void stop();

private:
    std::queue<CaptureWindow>  queue_;
    std::mutex                 mutex_;

    std::condition_variable    full_;
    std::condition_variable    empty_;
    int                        maxSize_;
    bool                       stopped_ = false;
};

#endif // WINDOW_QUEUE_H