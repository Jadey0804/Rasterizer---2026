#pragma once
const int grain = 4;
// Baseline: all false = no optimizations.
inline constexpr bool useLightOPT = 1;
inline constexpr bool useEdgeRaster = 1;
inline constexpr bool useRenderOPT = 1;
inline constexpr bool useScene1SharedMeshOPT = 0;
inline constexpr bool useBackfaceCulling = 1;
inline constexpr bool useSIMD = 1;

// Multi-thread raster config (only used when SIMD is enabled).
inline constexpr bool useMT = 1;
inline constexpr bool useSIMDMT = useSIMD && useMT;
inline constexpr size_t mtThreadCount = 4; // 0 = 自动；否则固定线程数（比如 1/2/4/8）
inline constexpr int mtTileW = 512;
inline constexpr int mtTileH = 384;




//scene1:
//baseline:244
//lightOPT:245
//
//renderOPT: 242
//useScene1SharedMeshOPT：238
//backface: 355
//simd:244
//MT：242
//SIMDMT: 346
//ALL TRUE except secen1sharedmeshopt: 1392
//thread (4) 512,384




//sence2:
//baseline:111
//LightOPT:116
//
//renderopt:115
//backface:153
//simd:116
//MT:111
//SIMDMT:264
//ALL TURE:868

