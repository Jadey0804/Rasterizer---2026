#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#include "BuildConfig.h"
#include <immintrin.h>


// Simple support class for a 2D vector
class vec2D {
public:
    float x, y;

    // Default constructor initializes both components to 0
    vec2D() { x = y = 0.f; };

    // Constructor initializes components with given values
    vec2D(float _x, float _y) : x(_x), y(_y) {}

    // Constructor initializes components from a vec4
    vec2D(vec4 v) {
        x = v[0];
        y = v[1];
    }

    // Display the vector components
    void display() { std::cout << x << '\t' << y << std::endl; }

    // Overloaded subtraction operator for vector subtraction
    vec2D operator- (vec2D& v) {
        vec2D q;
        q.x = x - v.x;
        q.y = y - v.y;
        return q;
    }
};

// Class representing a triangle for rendering purposes
class triangle {
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    colour col[3];     // Colors for each vertex of the triangle


    static inline float edgeFn(float ax, float ay, float bx, float by, float x, float y) {
		return(ay - by) * x + (bx - ax) * y + (ax * by -  bx*ay);
    }

    static inline bool isInside(float e01, float e12, float e20, float area2) {
        if (area2 > 0.0f)return(e01 >= 0.0f && e12 >= 0.0f && e20 >= 0.0f);
        else return(e01 <= 0.0f && e12 <= 0.0f && e20 <= 0.0f);
    }

	//------------------------- AVX2 version of the inside test -------------------------
    static inline __m256 insideMaskAVX2(__m256 e01, __m256 e12, __m256 e20, float area2) {
        const __m256 zero = _mm256_setzero_ps();
        if (area2 > 0.0f) {
            __m256 m01 = _mm256_cmp_ps(e01, zero, _CMP_GE_OQ);
            __m256 m12 = _mm256_cmp_ps(e12, zero, _CMP_GE_OQ);
            __m256 m20 = _mm256_cmp_ps(e20, zero, _CMP_GE_OQ);
            return _mm256_and_ps(_mm256_and_ps(m01, m12), m20);
        }
        else {
            __m256 m01 = _mm256_cmp_ps(e01, zero, _CMP_LE_OQ);
            __m256 m12 = _mm256_cmp_ps(e12, zero, _CMP_LE_OQ);
            __m256 m20 = _mm256_cmp_ps(e20, zero, _CMP_LE_OQ);
            return _mm256_and_ps(_mm256_and_ps(m01, m12), m20);
        }
    }


public:
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;

        // Calculate the 2D area of the triangle
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);
        area = std::fabs(e1.x * e2.y - e1.y * e2.x);
    }

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    float getC(vec2D v1, vec2D v2, vec2D p) {
        vec2D e = v2 - v1;
        vec2D q = p - v1;
        return q.y * e.x - q.x * e.y;
    }

    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
       
            alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
            beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
            gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;
        
        if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
        return true;
    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }

    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        if (useEdgeRaster) {
            draw_edgeRaster(renderer, L, ka, kd);
            return;
        }

        vec2D minV, maxV;
		//避免多线程多个triangle共享一个Light,不在像素里改L.omega_i,统一用局部lightDir;
        vec4 lightDir = L.omega_i;

        if (useLightOPT) {
			lightDir.normalise();
        }


        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (area < 1.f) return;

        // Iterate over the bounding box and check each pixel
        for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
            for (int x = (int)(minV.x); x < (int)ceil(maxV.x); x++) {
                float alpha, beta, gamma;

                // Check if the pixel lies inside the triangle
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {
                    // Interpolate color, depth, and normals
                    colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);

                    vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
					normal.normalise();
 
    

                    // Perform Z-buffer test and apply shading
                    if (renderer.zbuffer(x, y) > depth && depth > 0.001f) {
                        // typical shader begin
                        if (!useLightOPT) {
                            lightDir = L.omega_i;
							lightDir.normalise();

                        }
                        float dot = std::max(vec4::dot(lightDir, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka); // using kd instead of ka for ambient
                        // typical shader end

                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }

    void draw_edgeRaster(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;
		vec4 lightDir;
        

        if (useLightOPT) {
			lightDir = L.omega_i;
			lightDir.normalise();
        }

        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles (用 area2 版本也可以，但沿用你的逻辑)
        if (area < 1.f) return;

        const float x0 = v[0].p[0], y0 = v[0].p[1];
        const float x1 = v[1].p[0], y1 = v[1].p[1];
        const float x2 = v[2].p[0], y2 = v[2].p[1];

        // 用像素中心采样，避免边界漏点（如果你想和 baseline 完全一致，可以去掉 +0.5）
        int minX = (int)minV.x;
        int minY = (int)minV.y;
        int maxX = (int)ceil(maxV.x);
        int maxY = (int)ceil(maxV.y);

        // 防止越界（getBoundsWindow 已裁剪过，但这里再保险一次）
        minX = std::max(minX, 0);
        minY = std::max(minY, 0);
        maxX = std::min(maxX, (int)renderer.canvas.getWidth());
        maxY = std::min(maxY, (int)renderer.canvas.getHeight());

        // 计算有向面积 area2（注意：不是 fabs）
        const float area2 = edgeFn(x0, y0, x1, y1, x2, y2);
        if (area2 == 0.0f) return;
        const float invArea2 = 1.0f / area2; // 只做一次除法

        // 三条边的增量（x+1 加 dx；y+1 加 dy）
        const float e01_dx = (y0 - y1);
        const float e01_dy = (x1 - x0);

        const float e12_dx = (y1 - y2);
        const float e12_dy = (x2 - x1);

        const float e20_dx = (y2 - y0);
        const float e20_dy = (x0 - x2);

        // 光方向 normalize
       
        if (!useLightOPT) {
            lightDir.normalise();
        }
        else {
            lightDir.normalise();
        }

        // 起始采样点
        const float startX = (float)minX + 0.5;
        const float startY = (float)minY + 0.5;

        // 行起点的边函数值（只算一次）
        float e01_row = edgeFn(x0, y0, x1, y1, startX, startY);
        float e12_row = edgeFn(x1, y1, x2, y2, startX, startY);
        float e20_row = edgeFn(x2, y2, x0, y0, startX, startY);

        for (int y = minY; y < maxY; ++y) {
            float e01 = e01_row;
            float e12 = e12_row;
            float e20 = e20_row;

            float* zrow = renderer.zbuffer.rowPtr(y); // <-- 需要你实现 rowPtr

            int x = minX;

            // 8-wide lane offsets [0..7]
            const __m256 offsets = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
            const __m256 invA = _mm256_set1_ps(invArea2);
            const __m256 eps = _mm256_set1_ps(0.001f);
            const __m256 zero = _mm256_setzero_ps();

            // 顶点深度广播
            const __m256 z0 = _mm256_set1_ps(v[0].p[2]);
            const __m256 z1 = _mm256_set1_ps(v[1].p[2]);
            const __m256 z2 = _mm256_set1_ps(v[2].p[2]);

            // colour 广播（按你 colour 类字段改：这里假设 colour 有 .r .g .b）
            const __m256 c0r = _mm256_set1_ps(v[0].rgb.r), c1r = _mm256_set1_ps(v[1].rgb.r), c2r = _mm256_set1_ps(v[2].rgb.r);
            const __m256 c0g = _mm256_set1_ps(v[0].rgb.g), c1g = _mm256_set1_ps(v[1].rgb.g), c2g = _mm256_set1_ps(v[2].rgb.g);
            const __m256 c0b = _mm256_set1_ps(v[0].rgb.b), c1b = _mm256_set1_ps(v[1].rgb.b), c2b = _mm256_set1_ps(v[2].rgb.b);

            // normal 广播（vec4 用 [] 访问）
            const __m256 n0x = _mm256_set1_ps(v[0].normal[0]), n1x = _mm256_set1_ps(v[1].normal[0]), n2x = _mm256_set1_ps(v[2].normal[0]);
            const __m256 n0y = _mm256_set1_ps(v[0].normal[1]), n1y = _mm256_set1_ps(v[1].normal[1]), n2y = _mm256_set1_ps(v[2].normal[1]);
            const __m256 n0z = _mm256_set1_ps(v[0].normal[2]), n1z = _mm256_set1_ps(v[1].normal[2]), n2z = _mm256_set1_ps(v[2].normal[2]);

            // lightDir：你已经在 draw_edgeRaster 外面处理 useLightOPT，这里要求 lightDir 已经 normalise 好
            const __m256 lx = _mm256_set1_ps(lightDir[0]);
            const __m256 ly = _mm256_set1_ps(lightDir[1]);
            const __m256 lz = _mm256_set1_ps(lightDir[2]);

            // Light 颜色广播（按你 colour 字段改）
            const __m256 LLr = _mm256_set1_ps(L.L.r);
            const __m256 LLg = _mm256_set1_ps(L.L.g);
            const __m256 LLb = _mm256_set1_ps(L.L.b);

            const __m256 AMr = _mm256_set1_ps(L.ambient.r);
            const __m256 AMg = _mm256_set1_ps(L.ambient.g);
            const __m256 AMb = _mm256_set1_ps(L.ambient.b);

            const __m256 kdV = _mm256_set1_ps(kd);
            const __m256 kaV = _mm256_set1_ps(ka);

            // 主循环：每次 8 像素
            for (; x + 7 < maxX; x += 8) {
                // E(x+i)=E + i*dx
                __m256 E01 = _mm256_add_ps(_mm256_set1_ps(e01), _mm256_mul_ps(offsets, _mm256_set1_ps(e01_dx)));
                __m256 E12 = _mm256_add_ps(_mm256_set1_ps(e12), _mm256_mul_ps(offsets, _mm256_set1_ps(e12_dx)));
                __m256 E20 = _mm256_add_ps(_mm256_set1_ps(e20), _mm256_mul_ps(offsets, _mm256_set1_ps(e20_dx)));

                __m256 mInside = insideMaskAVX2(E01, E12, E20, area2);
                int insideBits = _mm256_movemask_ps(mInside);
                if (!insideBits) {
                    // 组步进：等价标量加 8 次
                    e01 += 8.0f * e01_dx;
                    e12 += 8.0f * e12_dx;
                    e20 += 8.0f * e20_dx;
                    continue;
                }

                // barycentric：w0=E12/area2,w1=E20/area2,w2=E01/area2
                __m256 w0 = _mm256_mul_ps(E12, invA);
                __m256 w1 = _mm256_mul_ps(E20, invA);
                __m256 w2 = _mm256_mul_ps(E01, invA);

                // depth
                __m256 depth = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(w0, z0), _mm256_mul_ps(w1, z1)),
                    _mm256_mul_ps(w2, z2)
                );

                __m256 zold = _mm256_loadu_ps(zrow + x);

                __m256 mZ = _mm256_cmp_ps(zold, depth, _CMP_GT_OQ);
                __m256 mEps = _mm256_cmp_ps(depth, eps, _CMP_GT_OQ);

                __m256 mPass = _mm256_and_ps(_mm256_and_ps(mInside, mZ), mEps);
                int passBits = _mm256_movemask_ps(mPass);
                if (!passBits) {
                    e01 += 8.0f * e01_dx;
                    e12 += 8.0f * e12_dx;
                    e20 += 8.0f * e20_dx;
                    continue;
                }

                // colour 插值
                __m256 r = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w0, c0r), _mm256_mul_ps(w1, c1r)), _mm256_mul_ps(w2, c2r));
                __m256 g = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w0, c0g), _mm256_mul_ps(w1, c1g)), _mm256_mul_ps(w2, c2g));
                __m256 b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w0, c0b), _mm256_mul_ps(w1, c1b)), _mm256_mul_ps(w2, c2b));

                // clamp 0..1
                const __m256 one = _mm256_set1_ps(1.0f);
                r = _mm256_min_ps(_mm256_max_ps(r, zero), one);
                g = _mm256_min_ps(_mm256_max_ps(g, zero), one);
                b = _mm256_min_ps(_mm256_max_ps(b, zero), one);

                // normal 插值 + normalize
                __m256 nx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w0, n0x), _mm256_mul_ps(w1, n1x)), _mm256_mul_ps(w2, n2x));
                __m256 ny = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w0, n0y), _mm256_mul_ps(w1, n1y)), _mm256_mul_ps(w2, n2y));
                __m256 nz = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w0, n0z), _mm256_mul_ps(w1, n1z)), _mm256_mul_ps(w2, n2z));

                __m256 len2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)), _mm256_mul_ps(nz, nz));
                __m256 invLen = _mm256_rsqrt_ps(len2);

                // 牛顿迭代一次（精度更稳）
                const __m256 half = _mm256_set1_ps(0.5f);
                const __m256 one5 = _mm256_set1_ps(1.5f);
                invLen = _mm256_mul_ps(invLen,
                    _mm256_sub_ps(one5,
                        _mm256_mul_ps(half, _mm256_mul_ps(len2, _mm256_mul_ps(invLen, invLen)))));

                nx = _mm256_mul_ps(nx, invLen);
                ny = _mm256_mul_ps(ny, invLen);
                nz = _mm256_mul_ps(nz, invLen);

                // dot + clamp
                __m256 dot = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(nx, lx), _mm256_mul_ps(ny, ly)), _mm256_mul_ps(nz, lz));
                dot = _mm256_max_ps(dot, zero);

                // shading： (c*kd)*(L.L*dot) + ambient*ka
                __m256 outR = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(r, kdV), LLr), dot), _mm256_mul_ps(AMr, kaV));
                __m256 outG = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(g, kdV), LLg), dot), _mm256_mul_ps(AMg, kaV));
                __m256 outB = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(b, kdV), LLb), dot), _mm256_mul_ps(AMb, kaV));

                outR = _mm256_min_ps(_mm256_max_ps(outR, zero), one);
                outG = _mm256_min_ps(_mm256_max_ps(outG, zero), one);
                outB = _mm256_min_ps(_mm256_max_ps(outB, zero), one);

                // z 写回（masked store）
                __m256i mPassI = _mm256_castps_si256(mPass);
                _mm256_maskstore_ps(zrow + x, mPassI, depth);

                // 颜色写回：你目前只能用 draw()，因此按 mask 标量写（计算 SIMD，写回标量）
                alignas(32) float rr[8], gg[8], bb[8];
                _mm256_store_ps(rr, outR);
                _mm256_store_ps(gg, outG);
                _mm256_store_ps(bb, outB);

                for (int lane = 0; lane < 8; ++lane) {
                    if (passBits & (1 << lane)) {
                        unsigned char R = (unsigned char)(rr[lane] * 255.0f);
                        unsigned char G = (unsigned char)(gg[lane] * 255.0f);
                        unsigned char B = (unsigned char)(bb[lane] * 255.0f);
                        renderer.canvas.draw(x + lane, y, R, G, B);
                    }
                }

                // 组步进
                e01 += 8.0f * e01_dx;
                e12 += 8.0f * e12_dx;
                e20 += 8.0f * e20_dx;
            }

            // tail：剩余像素用你原本标量版本
            for (; x < maxX; ++x) {
                if (isInside(e01, e12, e20, area2)) {
                    const float w0 = e12 * invArea2;
                    const float w1 = e20 * invArea2;
                    const float w2 = e01 * invArea2;

                    const float beta = w0;
                    const float gamma = w1;
                    const float alpha = w2;

                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);

                    if ((renderer.zbuffer(x, y) > depth && depth > 0.001f)) {
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();

                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                        normal.normalise();

                        if (!useLightOPT) {
                            lightDir = L.omega_i;
                            lightDir.normalise();
                        }

                        float dotS = std::max(vec4::dot(lightDir, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dotS) + (L.ambient * ka);

                        unsigned char R, G, B;
                        a.toRGB(R, G, B);
                        renderer.canvas.draw(x, y, R, G, B);
                        renderer.zbuffer(x, y) = depth;
                    }
                }

                e01 += e01_dx;
                e12 += e12_dx;
                e20 += e20_dx;
            }

            e01_row += e01_dy;
            e12_row += e12_dy;
            e20_row += e20_dy;
        }

    }


    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    void getBounds(vec2D& minV, vec2D& maxV) {
        minV = vec2D(v[0].p);
        maxV = vec2D(v[0].p);
        for (unsigned int i = 1; i < 3; i++) {
            minV.x = std::min(minV.x, v[i].p[0]);
            minV.y = std::min(minV.y, v[i].p[1]);
            maxV.x = std::max(maxV.x, v[i].p[0]);
            maxV.y = std::max(maxV.y, v[i].p[1]);
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) {
        getBounds(minV, maxV);
        minV.x = std::max(minV.x, static_cast<float>(0));
        minV.y = std::max(minV.y, static_cast<float>(0));
        maxV.x = std::min(maxV.x, static_cast<float>(canvas.getWidth()));
        maxV.y = std::min(maxV.y, static_cast<float>(canvas.getHeight()));
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas) {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        for (int y = (int)minV.y; y < (int)maxV.y; y++) {
            for (int x = (int)minV.x; x < (int)maxV.x; x++) {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() {
        for (unsigned int i = 0; i < 3; i++) {
            v[i].p.display();
        }
        std::cout << std::endl;
    }
};
