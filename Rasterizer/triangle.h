#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#include "BuildConfig.h"

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

            for (int x = minX; x < maxX; ++x) {
                if (isInside(e01, e12, e20, area2)) {
                    // barycentric：w0 对 v0, w1 对 v1, w2 对 v2
                    // 常用对应：w0 = E12/area2, w1 = E20/area2, w2 = E01/area2
                    const float w0 = e12 * invArea2;
                    const float w1 = e20 * invArea2;
                    const float w2 = e01 * invArea2;

                    const float beta = w0;
                    const float gamma = w1;
                    const float alpha = w2;

                    colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();

                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);

                    vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
					normal.normalise();

                    if (renderer.zbuffer(x, y) > depth && depth > 0.001f) {
                       
                        if (!useLightOPT) {
                            lightDir = L.omega_i;
                            lightDir.normalise();
						}
                        
                        float dot = std::max(vec4::dot(lightDir, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka);

                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }

                // x+1：边函数增量更新（只加法）
                e01 += e01_dx;
                e12 += e12_dx;
                e20 += e20_dx;
            }

            // y+1：行起点增量更新
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
