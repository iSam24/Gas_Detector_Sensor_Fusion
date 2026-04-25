#include "windowQueue.h"

// Called by capture thread, blocks if queue is full
void WindowQueue::push(CaptureWindow window) {
    std::unique_lock<std::mutex> lock(mutex_);  // lock mutex

    // Tells the current thread to wait until the queue is not full, or until we’ve been told to stop
    // unlocks the mutex while sleeping
    // wakes up when full_ is notified
    // re checks condition
    // proceeds when condition is true
    full_.wait(lock, [this] {return queue_.size() < maxSize_ || stopped_;});

    if (stopped_) return;

    queue_.push(std::move(window));
    
    empty_.notify_one(); // notify the waiting inference thread (consumer) that data is available
}

// Get data from the queue
std::optional<CaptureWindow> WindowQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);  // lock mutex

    // wait until queue is not empty, or we've been told to stop
    // unlocks the mutex while sleeping
    // wakes up when empty_ is notified
    // re checks condition
    // proceeds when condition is true
    empty_.wait(lock, [this] {return !queue_.empty() || stopped_;});

    if (queue_.empty()) return std::nullopt;  // stopped + empty

    auto window = std::move(queue_.front());

    queue_.pop();

    full_.notify_one(); // notify the waiting capture thread (producer) that a window has been consumed

    return window;
}

void WindowQueue::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    empty_.notify_all();
    full_.notify_all(); 
}
