#pragma once

const int grain = 4;

const bool useLightOPT = true;
const bool useEdgeRaster = true;
//const bool useFastNormalizeOPT = true;
const bool useRenderOPT = true;
const bool useScene1SharedMeshOPT = false;
const bool useBackfaceCulling = true;
//const bool useFrustumCullingClip = true;
const bool useSIMD = true;

// Multi-thread raster config
inline constexpr bool  useMT = true;          // 总开关：true=tile并行 raster
inline constexpr size_t mtThreadCount = 4;    // 0 = 自动；否则固定线程数（比如 1/2/4/8）
inline constexpr int   mtTileW = 512;          // tile 宽
inline constexpr int   mtTileH = 384;          // tile 高
