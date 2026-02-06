// Example integration idea for your rasterizer:
// Split screen into tiles (e.g., 32x16 pixels) and run tiles in parallel.
// Each tile has its own bounding box; avoid false sharing by writing to separate rows/tiles.

#include "ThreadPool.h"

void rasterizeFrame(ThreadPool& pool, int width, int height) {
    const int tileW = 32, tileH = 16;
    const int tilesX = (width + tileW - 1) / tileW;
    const int tilesY = (height + tileH - 1) / tileH;

    pool.parallel_for(0, (size_t)(tilesX * tilesY), [&](size_t t) {
        int ty = (int)(t / tilesX);
        int tx = (int)(t % tilesX);
        int x0 = tx * tileW;
        int y0 = ty * tileH;
        int x1 = std::min(x0 + tileW, width);
        int y1 = std::min(y0 + tileH, height);

        // rasterizeTrianglesIntoRect(x0, y0, x1, y1);
        }, /*grain=*/1);
}
