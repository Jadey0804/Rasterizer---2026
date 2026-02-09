#pragma once
#include <cmath>
#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <vector>
#include <chrono>

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
#include"ThreadPool.h"

struct Scene3Object {
    Mesh* mesh = nullptr;
    matrix world = matrix::makeIdentity();
    vec4 center = vec4(0.f, 0.f, 0.f, 1.f);
    float radius = 1.0f;
    vec4 rotSpeed = vec4(0.f, 0.f, 0.f, 0.f);
};
