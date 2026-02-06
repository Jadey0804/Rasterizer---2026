#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <vector>
#include <chrono>

#include <cmath>
#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"
#include "Timer.h"

#include "BuildConfig.h"
#include <immintrin.h>
#include <vector>
#include"ThreadPool.h"

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.

static inline void getMatrixElems(const matrix& M,
    float& m00, float& m01, float& m02, float& m03,
    float& m10, float& m11, float& m12, float& m13,
    float& m20, float& m21, float& m22, float& m23,
    float& m30, float& m31, float& m32, float& m33)
{
    m00 = M.get(0, 0); m01 = M.get(0, 1); m02 = M.get(0, 2); m03 = M.get(0, 3);
    m10 = M.get(1, 0); m11 = M.get(1, 1); m12 = M.get(1, 2); m13 = M.get(1, 3);
    m20 = M.get(2, 0); m21 = M.get(2, 1); m22 = M.get(2, 2); m23 = M.get(2, 3);
    m30 = M.get(3, 0); m31 = M.get(3, 1); m32 = M.get(3, 2); m33 = M.get(3, 3);
}


void renderOPT_AVX2(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, ThreadPool* pool) {

    const matrix P = renderer.perspective * camera * mesh->world;

    float m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33;
    getMatrixElems(P, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);

    const size_t N = mesh->vertices.size();
	//static thread_local std::vector<Vertex> tv;
    std::vector<Vertex> tv;
    tv.resize(N);

    // SOA input
    static thread_local std::vector<float> px, py, pz, pw;
    px.resize(N); py.resize(N); pz.resize(N); pw.resize(N);

    for (size_t i = 0; i < N; ++i) {
        const vec4& p = mesh->vertices[i].p;
        px[i] = p[0]; 
        py[i] = p[1]; 
        pz[i] = p[2];
        pw[i] = p[3]; // 若你的 w 恒 1，可直接 pw[i]=1.f
    }

    const float W = (float)renderer.canvas.getWidth();
    const float H = (float)renderer.canvas.getHeight();

    const __m256 m00v = _mm256_set1_ps(m00), m01v = _mm256_set1_ps(m01), m02v = _mm256_set1_ps(m02), m03v = _mm256_set1_ps(m03);
    const __m256 m10v = _mm256_set1_ps(m10), m11v = _mm256_set1_ps(m11), m12v = _mm256_set1_ps(m12), m13v = _mm256_set1_ps(m13);
    const __m256 m20v = _mm256_set1_ps(m20), m21v = _mm256_set1_ps(m21), m22v = _mm256_set1_ps(m22), m23v = _mm256_set1_ps(m23);
    const __m256 m30v = _mm256_set1_ps(m30), m31v = _mm256_set1_ps(m31), m32v = _mm256_set1_ps(m32), m33v = _mm256_set1_ps(m33);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 Wv = _mm256_set1_ps(W);
    const __m256 Hv = _mm256_set1_ps(H);


    // 先把 tv[i] 初始化好：拷贝 rgb，并把 normal 提前处理（标量）
    for (size_t i = 0; i < N; ++i) {
        tv[i] = mesh->vertices[i]; // 拷贝 rgb/normal/其它属性（不拷贝也行，但最省事）

        // normal 变换 & normalize：移到 SIMD loop 外
        tv[i].normal = mesh->world * mesh->vertices[i].normal;
        tv[i].normal.normalise();
    }

    size_t i = 0;
	// 
    for (; i + 7 < N; i += 8) {
		// Matrix-vector multiplication using AVX2
        __m256 x = _mm256_loadu_ps(&px[i]);
        __m256 y = _mm256_loadu_ps(&py[i]);
        __m256 z = _mm256_loadu_ps(&pz[i]);
        __m256 w = _mm256_loadu_ps(&pw[i]);

        // clip = M * [x y z w]
        __m256 cx = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(m00v, x), _mm256_mul_ps(m01v, y)),
            _mm256_add_ps(_mm256_mul_ps(m02v, z), _mm256_mul_ps(m03v, w)));

        __m256 cy = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(m10v, x), _mm256_mul_ps(m11v, y)),
            _mm256_add_ps(_mm256_mul_ps(m12v, z), _mm256_mul_ps(m13v, w)));

        __m256 cz = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(m20v, x), _mm256_mul_ps(m21v, y)),
            _mm256_add_ps(_mm256_mul_ps(m22v, z), _mm256_mul_ps(m23v, w)));

        __m256 cw = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(m30v, x), _mm256_mul_ps(m31v, y)),
            _mm256_add_ps(_mm256_mul_ps(m32v, z), _mm256_mul_ps(m33v, w)));

		// Divide by W (perspective divide)
        // invW = 1/cw
        __m256 invW = _mm256_div_ps(one, cw);

		// Map to NDC space
        // ndc
        __m256 ndcX = _mm256_mul_ps(cx, invW);
        __m256 ndcY = _mm256_mul_ps(cy, invW);
        __m256 ndcZ = _mm256_mul_ps(cz, invW);

        // screen mapping: sx=(ndcX+1)*0.5*W, sy=H - (ndcY+1)*0.5*H
        __m256 sx = _mm256_mul_ps(_mm256_mul_ps(_mm256_add_ps(ndcX, one), half), Wv);
        __m256 sy = _mm256_sub_ps(Hv, _mm256_mul_ps(_mm256_mul_ps(_mm256_add_ps(ndcY, one), half), Hv));

        // store back to tv (AOS)
        alignas(32) float sxArr[8], syArr[8], szArr[8];
        _mm256_store_ps(sxArr, sx);
        _mm256_store_ps(syArr, sy);
        _mm256_store_ps(szArr, ndcZ);

        //--------------------------
        for (int k = 0; k < 8; ++k) {
            tv[i + k].p[0] = sxArr[k];
            tv[i + k].p[1] = syArr[k];
            tv[i + k].p[2] = szArr[k];
        }

    }

    // tail scalar
    for (; i < N; ++i) {
        vec4 p = P * mesh->vertices[i].p;
        p.divideW();

        p[0] = (p[0] + 1.f) * 0.5f * W;
        p[1] = (p[1] + 1.f) * 0.5f * H;
        p[1] = H - p[1];

        tv[i].p[0] = p[0];
        tv[i].p[1] = p[1];
        tv[i].p[2] = p[2];
    }


    // --- after tv is ready ---

    const int Wi = (int)renderer.canvas.getWidth();
    const int Hi = (int)renderer.canvas.getHeight();

    const int tileW = mtTileW;
    const int tileH = mtTileH;
    const int tilesX = (Wi + tileW - 1) / tileW;
    const int tilesY = (Hi + tileH - 1) / tileH;
    const int tileCount = tilesX * tilesY;

    // --- Triangle binning: tile -> list of triangle indices ---
    static std::vector<std::vector<uint32_t>> tileBins;
    tileBins.resize(tileCount);
    for (auto& b : tileBins) b.clear(); // 每帧清空桶（保留 capacity）



    // 单线程 fallback（用于对比/或者关掉 MT）
    if (!useMT || pool == nullptr || pool->size() <= 1) {
        for (const triIndices& ind : mesh->triangles) {
            const Vertex& v0 = tv[ind.v[0]];
            const Vertex& v1 = tv[ind.v[1]];
            const Vertex& v2 = tv[ind.v[2]];
            if (fabs(v0.p[2]) > 1.0f || fabs(v1.p[2]) > 1.0f || fabs(v2.p[2]) > 1.0f) continue;

            triangle tri(v0, v1, v2);
            tri.draw(renderer, L, mesh->ka, mesh->kd, nullptr); // 无 scissor
        }
        return;
    }






    // MT tile raster
    pool->parallel_for(0, (size_t)tileCount, [&](size_t tid) {
        const int tx = (int)(tid % tilesX);
        const int ty = (int)(tid / tilesX);

        Scissor s{
            tx * tileW,
            ty * tileH,
            std::min(Wi, tx * tileW + tileW),
            std::min(Hi, ty * tileH + tileH)
        };

        for (const triIndices& ind : mesh->triangles) {
            const Vertex& v0 = tv[ind.v[0]];
            const Vertex& v1 = tv[ind.v[1]];
            const Vertex& v2 = tv[ind.v[2]];

            if (fabs(v0.p[2]) > 1.0f || fabs(v1.p[2]) > 1.0f || fabs(v2.p[2]) > 1.0f) continue;

            triangle tri(v0, v1, v2);
            tri.draw(renderer, L, mesh->ka, mesh->kd, &s);
        }
        }, grain);

}



void renderOPT(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {

 //   if (useSIMD) {
 //       renderOPT_AVX2(renderer, mesh, camera, L, pool);
 //       return;
	//}
    
    // Combine perspective, camera, and world transformations for the mesh
    const matrix p = renderer.perspective * camera * mesh->world;

    // --- OPT1: Transform every vertex once per mesh (per frame) ---
    // Cache transformed vertices (screen-space position + transformed normal + rgb)
    std::vector<Vertex> tv;
    tv.resize(mesh->vertices.size());

    const float w = static_cast<float>(renderer.canvas.getWidth());
    const float h = static_cast<float>(renderer.canvas.getHeight());

    for (size_t i = 0; i < mesh->vertices.size(); ++i) {
        Vertex out = mesh->vertices[i];  // copy rgb/normal/position (object space)

        // Position: object -> clip
        out.p = p * mesh->vertices[i].p;
        out.p.divideW(); // NDC in [-1,1]

        // Normal: object -> world (same as your original code)
        out.normal = mesh->world * mesh->vertices[i].normal;
        out.normal.normalise();

        // NDC -> screen
        out.p[0] = (out.p[0] + 1.f) * 0.5f * w;
        out.p[1] = (out.p[1] + 1.f) * 0.5f * h;
        out.p[1] = h - out.p[1];

        // rgb already copied via Vertex out = mesh->vertices[i]
        tv[i] = out;
    }

    // --- Triangle loop: only gather 3 cached vertices and draw ---
	//-----------use Backface Culling----------------
    if (useBackfaceCulling) {
        for (triIndices& ind : mesh->triangles) {
            const Vertex& v0 = tv[ind.v[0]];
            const Vertex& v1 = tv[ind.v[1]];
            const Vertex& v2 = tv[ind.v[2]];

            // 原本的 z check
            if (fabs(v0.p[2]) > 1.0f || fabs(v1.p[2]) > 1.0f || fabs(v2.p[2]) > 1.0f) continue;

            // -------- Back-face culling (screen-space) --------
            if (useBackfaceCulling) {
                float x0 = v0.p[0], y0 = v0.p[1];
                float x1 = v1.p[0], y1 = v1.p[1];
                float x2 = v2.p[0], y2 = v2.p[1];

                float area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);

                if (area2 <= 0.0f) continue;
            }
            // -----------------------------------------------

            triangle tri(v0, v1, v2);
            tri.draw(renderer, L, mesh->ka, mesh->kd);
        }

    }

    else {

        for (triIndices& ind : mesh->triangles) {
            const Vertex& v0 = tv[ind.v[0]];
            const Vertex& v1 = tv[ind.v[1]];
            const Vertex& v2 = tv[ind.v[2]];

            // Clip triangles with Z-values outside [-1, 1]
            if (fabs(v0.p[2]) > 1.0f || fabs(v1.p[2]) > 1.0f || fabs(v2.p[2]) > 1.0f) continue;

            triangle tri(v0, v1, v2);
            tri.draw(renderer, L, mesh->ka, mesh->kd);
        }
    }
}


void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {

    if (useRenderOPT) {
        renderOPT(renderer, mesh, camera, L );
        return;
    }

    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;


    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal; 
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        if(useBackfaceCulling) {
            // -------- Back-face culling (screen-space signed area) --------
            if (useBackfaceCulling) {
                const float x0 = t[0].p[0], y0 = t[0].p[1];
                const float x1 = t[1].p[0], y1 = t[1].p[1];
                const float x2 = t[2].p[0], y2 = t[2].p[1];

                // area2 = cross((p1-p0),(p2-p0)) in 2D
                const float area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);

                if (area2 <= 0.0f) continue;
            }
            // ----------------------------------------------------------------
		}
        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}

// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
	ThreadPool pool;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // camera is just a matrix
    matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
   // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        // Render each object in the scene
        for (auto& m : scene)
            render(renderer, m, camera, L);

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    FrameTimer timer;
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f),
             colour(1.0f, 1.0f, 1.0f),
             colour(0.2f, 0.2f, 0.2f) };

    const int W = (int)renderer.canvas.getWidth();
    const int H = (int)renderer.canvas.getHeight();

    const int tileW = mtTileW;
    const int tileH = mtTileH;
    const int tilesX = (W + tileW - 1) / tileW;
    const int tilesY = (H + tileH - 1) / tileH;
    const int tileCount = tilesX * tilesY;

    size_t threads = mtThreadCount;
    if (threads == 0) {
        threads = std::min((size_t)std::thread::hardware_concurrency(), (size_t)tileCount);
        if (threads == 0) threads = 1;
    }

    std::unique_ptr<ThreadPool> pool;
    if (useMT && threads > 1) pool = std::make_unique<ThreadPool>(threads);


    float zoffset = 8.0f;
    float step = -0.1f;


    std::vector<Mesh*> scene;        // baseline
    Mesh cube;                        // optimized
    std::vector<matrix> worlds;      // optimized

    if (!useScene1SharedMeshOPT) {
        // -------- baseline：原始 scene1（40 个 Mesh） --------
        for (unsigned int i = 0; i < 20; i++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            m->world = matrix::makeTranslation(-2.0f, 0.0f, -3.f * i)
                * makeRandomRotation();
            scene.push_back(m);

            m = new Mesh();
            *m = Mesh::makeCube(1.f);
            m->world = matrix::makeTranslation(2.0f, 0.0f, -3.f * i)
                * makeRandomRotation();
            scene.push_back(m);
        }
    }
    else {
        // -------- optimized：共享一个 cube mesh --------
        cube = Mesh::makeCube(1.f);
        worlds.reserve(40);

        for (unsigned int i = 0; i < 20; i++) {
            worlds.push_back(
                matrix::makeTranslation(-2.0f, 0.0f, -3.f * i)
                * makeRandomRotation()
            );
            worlds.push_back(
                matrix::makeTranslation(2.0f, 0.0f, -3.f * i)
                * makeRandomRotation()
            );
        }
    }


    while (!timer.finished()) {
        timer.beginFrame();
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) step *= -1.f;

        if (!useScene1SharedMeshOPT) {
            // -------- baseline render --------
            scene[0]->world = scene[0]->world
                * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
            scene[1]->world = scene[1]->world
                * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

            for (auto& m : scene) {
                if (useRenderOPT && useSIMD) {
                    renderOPT_AVX2(renderer, m, camera, L, pool ? pool.get() : nullptr);
                }
                else if (useRenderOPT) {
                    renderOPT(renderer, m, camera, L);
                }
                else {
                    render(renderer, m, camera, L);
                }
            }

        }
        else {
            // -------- optimized render --------
            worlds[0] = worlds[0]
                * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
            worlds[1] = worlds[1]
                * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

            for (const auto& W : worlds) {
                cube.world = W;

                if (useRenderOPT && useSIMD) {
                    renderOPT_AVX2(renderer, &cube, camera, L, pool ? pool.get() : nullptr);
                }
                else if (useRenderOPT) {
                    renderOPT(renderer, &cube, camera, L);
                }
                else {
                    render(renderer, &cube, camera, L);
                }
            }

        }

        renderer.present();
        timer.endFrame();
    }


    if (!useScene1SharedMeshOPT) {
        for (auto& m : scene) delete m;
    }

    std::cout << (useScene1SharedMeshOPT ? "[OPT]" : "[BASE]")
        << " FPS: " << timer.averageFPS()
        << " ; " << timer.averageFrameTimeMs() << " ms\n";
}


// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    FrameTimer timer;
    Renderer renderer;
	ThreadPool pool;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;

    while (!timer.finished()) {
        timer.beginFrame();
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                //end = std::chrono::high_resolution_clock::now();
                //std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                //start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
            render(renderer, m, camera, L);
        renderer.present();

        timer.endFrame();
    }

    for (auto& m : scene)
        delete m;

    std::cout << timer.averageFPS() << "; " << timer.averageFrameTimeMs() << " ms" << std::endl;


}

// Entry point of the application
// No input variables
int main() {
    // Uncomment the desired scene function to run
    scene1();
    //scene2();
    //sceneTest(); 
    

    return 0;
}