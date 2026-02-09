#pragma once
const int grain = 4;
// Baseline: all false = no optimizations.
inline constexpr bool useLightOPT =1;
inline constexpr bool useEdgeRaster = 0;
inline constexpr bool useRenderOPT = 0;
inline constexpr bool useScene1SharedMeshOPT = 0;
inline constexpr bool useBackfaceCulling = 0;
inline constexpr bool useSIMD = 0;

// Multi-thread raster config (only used when SIMD is enabled).
inline constexpr bool useMT = 0;
inline constexpr bool useSIMDMT = useSIMD && useMT;
inline constexpr size_t mtThreadCount = 0; // 0 = 自动；否则固定线程数（比如 1/2/4/8）
inline constexpr int mtTileW = 1024;
inline constexpr int mtTileH = 768;

// Scene3-only coarse culling switch (helps heavy scenes).
inline constexpr bool useScene3FrustumCull = 1;


//======================= scene1 =========================
//baseline:244
//lightOPT:260
//useEdgeRaster：738
//renderOPT: 242
//useScene1SharedMeshOPT：238
//backface: 355
//simd:244
//MT：242
//SIMDMT: 346

//ALL TRUE except secen1sharedmeshopt: 
//thread (1) 1024,768  1028
//thread (2) 512,768   1299
//thread (4) 512,384   1392//max
//thread (6) 512,256   1134
//thread (8) 256,384   1016
//thread (12) 256,256  915



//======================= scene2 ============================
//baseline:111
//LightOPT:116
//useEdgeRaster：326
//renderopt:115
//backface:153
//simd:116
//MT:111
//SIMDMT:264

//ALL TURE:
//thread (1) 1024,768  490
//thread (2) 512,768   683
//thread (4) 512,384   902//max
//thread (6) 512,256   636
//thread (8) 256,384   638
//thread (12) 256,256  724




//======================= scene3 ============================
//baseline:44
//LightOPT:46
//useEdgeRaster：48
//renderopt:62
//backface:59
//simd:59
//MT:
//SIMDMT:43
//useScene3FrustumCull：46

//ALL TURE:
//thread (1) 1024,768  110//max
//thread (2) 512,768   86
//thread (4) 512,384   92
//thread (6) 512,256   84
//thread (8) 256,384   84
//thread (12) 256,256  63
