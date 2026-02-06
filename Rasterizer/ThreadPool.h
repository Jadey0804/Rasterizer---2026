// ThreadPool.h (C++20) — efficient fixed-size thread pool
// - Per-thread local deques + global injector queue
// - Work stealing (LIFO locally, FIFO stealing) for good cache locality
// - Low overhead: spin-then-sleep, minimal locking, no busy-wait burning
//
// Usage:
//   ThreadPool pool; // default: hardware_concurrency()
//   auto fut = pool.submit([]{ return 123; });
//   int v = fut.get();
//
//   pool.parallel_for(0, N, [&](size_t i){ ... }, /*grain=*/256);

#pragma once
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t threadCount = std::thread::hardware_concurrency())
        : stopping_(false)
    {
        if (threadCount == 0) threadCount = 1;
        queues_.resize(threadCount);
        threads_.reserve(threadCount);

        for (std::size_t i = 0; i < threadCount; ++i) {
            threads_.emplace_back([this, i] { workerLoop(i); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() { shutdown(); }

    void shutdown() {
        bool expected = false;
        if (!stopping_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;

        // Wake all workers
        {
            std::lock_guard<std::mutex> lk(cvMutex_);
        }
        cv_.notify_all();

        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
        threads_.clear();
    }

    std::size_t size() const noexcept { return threads_.size(); }

    // Submit any callable; returns future<R>
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;

        auto taskPtr = std::make_shared<std::packaged_task<R()>>(
            [fn = std::forward<F>(f),
            tup = std::make_tuple(std::forward<Args>(args)...)]() mutable -> R {
                return std::apply(std::move(fn), std::move(tup));
            });

        std::future<R> fut = taskPtr->get_future();
        enqueue([taskPtr]() mutable { (*taskPtr)(); });
        return fut;
    }

    // Fire-and-forget (no future)
    void enqueue(std::function<void()> task) {
        if (stopping_.load(std::memory_order_acquire)) return;

        // Prefer local queue if called from a worker thread
        if (auto idx = tlsWorkerIndex_; idx.has_value()) {
            auto& q = queues_[*idx];
            {
                std::lock_guard<std::mutex> lk(q.m);
                q.local.emplace_back(std::move(task)); // LIFO for locality
            }
            pending_.fetch_add(1, std::memory_order_release);
            cv_.notify_one();
            return;
        }

        // Otherwise push into global injector
        {
            std::lock_guard<std::mutex> lk(global_.m);
            global_.q.emplace_back(std::move(task));
        }
        pending_.fetch_add(1, std::memory_order_release);
        cv_.notify_one();
    }

    // Wait until all currently queued work completes
    void waitIdle() {
        // Simple: wait until pending_ reaches 0
        while (pending_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }

    // Parallel for [begin, end)
    template <class Fn>
    void parallel_for(std::size_t begin, std::size_t end, Fn&& fn, std::size_t grain = 256) {
        if (end <= begin) return;
        if (grain == 0) grain = 1;

        const std::size_t n = end - begin;
        const std::size_t blocks = (n + grain - 1) / grain;

        std::atomic<std::size_t> done{ 0 };
        std::promise<void> allDone;
        auto allDoneFut = allDone.get_future();

        for (std::size_t b = 0; b < blocks; ++b) {
            const std::size_t b0 = begin + b * grain;
            const std::size_t b1 = std::min(end, b0 + grain);

            enqueue([&, b0, b1]() mutable {
                for (std::size_t i = b0; i < b1; ++i) fn(i);
                if (done.fetch_add(1, std::memory_order_acq_rel) + 1 == blocks) {
                    allDone.set_value();
                }
                });
        }

        allDoneFut.wait();
    }

private:
    struct alignas(64) LocalQueue {
        std::mutex m;
        std::deque<std::function<void()>> local;
    };

    struct alignas(64) GlobalQueue {
        std::mutex m;
        std::deque<std::function<void()>> q;
    };

	inline static thread_local std::optional<std::size_t> tlsWorkerIndex_ = std::nullopt;

    std::vector<std::thread> threads_;
    std::deque<LocalQueue> queues_;
    GlobalQueue global_;

    std::atomic<bool> stopping_;
    std::atomic<std::size_t> pending_{ 0 };

    std::condition_variable cv_;
    std::mutex cvMutex_; // for condition_variable wait

    bool tryPopLocal(std::size_t self, std::function<void()>& out) {
        auto& q = queues_[self];
        std::lock_guard<std::mutex> lk(q.m);
        if (q.local.empty()) return false;
        out = std::move(q.local.back()); // LIFO
        q.local.pop_back();
        return true;
    }

    bool trySteal(std::size_t self, std::function<void()>& out) {
        // Steal from other workers FIFO (front) to reduce contention with their LIFO
        const std::size_t n = queues_.size();
        if (n <= 1) return false;

        // Simple round-robin starting point
        std::size_t start = (self + 1) % n;
        for (std::size_t k = 0; k < n - 1; ++k) {
            std::size_t victim = (start + k) % n;
            auto& q = queues_[victim];
            std::lock_guard<std::mutex> lk(q.m);
            if (!q.local.empty()) {
                out = std::move(q.local.front()); // FIFO steal
                q.local.pop_front();
                return true;
            }
        }
        return false;
    }

    bool tryPopGlobal(std::function<void()>& out) {
        std::lock_guard<std::mutex> lk(global_.m);
        if (global_.q.empty()) return false;
        out = std::move(global_.q.front()); // FIFO injector
        global_.q.pop_front();
        return true;
    }

    void workerLoop(std::size_t index) {
        tlsWorkerIndex_ = index;

        // Spin-then-sleep parameters
        constexpr int kSpinIters = 200;

        while (!stopping_.load(std::memory_order_acquire)) {
            std::function<void()> task;

            bool got = tryPopLocal(index, task) || tryPopGlobal(task) || trySteal(index, task);
            if (got) {
                task();
                pending_.fetch_sub(1, std::memory_order_release);
                continue;
            }

            // Brief spin to catch bursts without kernel sleep
            for (int i = 0; i < kSpinIters; ++i) {
                if (stopping_.load(std::memory_order_acquire)) break;
                if (pending_.load(std::memory_order_acquire) != 0) break;
                std::this_thread::yield();
            }

            if (stopping_.load(std::memory_order_acquire)) break;

            // Sleep until notified or new work arrives
            std::unique_lock<std::mutex> lk(cvMutex_);
            cv_.wait(lk, [&] {
                return stopping_.load(std::memory_order_acquire) ||
                    pending_.load(std::memory_order_acquire) != 0;
                });
        }

        tlsWorkerIndex_.reset();
    }
};



