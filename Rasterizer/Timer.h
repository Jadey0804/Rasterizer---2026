#pragma once
#include <chrono>
#include <cstdint>

class FrameTimer {
public:
    FrameTimer(int warmupFrames = 2000, int measureFrames = 10000)
        : warmupFrames(warmupFrames),
        measureFrames(measureFrames) {
    }

    void beginFrame() {
        frameStart = clock::now();
    }

    void endFrame() {
        auto frameEnd = clock::now();
        double ms = std::chrono::duration<double, std::milli>(
            frameEnd - frameStart).count();

        totalFrames++;

        // skip warmup frames
        if (totalFrames <= warmupFrames)
            return;

        if (measuredFrames < measureFrames) {
            accumulatedMs += ms;
            measuredFrames++;
        }
    }

    bool finished() const {
        return measuredFrames >= measureFrames;
    }

    double averageFrameTimeMs() const {
        return accumulatedMs / measuredFrames;
    }

    double averageFPS() const {
        return 1000.0 / averageFrameTimeMs();
    }

private:
    using clock = std::chrono::high_resolution_clock;

    int warmupFrames = 0;
    int measureFrames = 0;

    int totalFrames = 0;
    int measuredFrames = 0;

    double accumulatedMs = 0.0;
    clock::time_point frameStart;
};

