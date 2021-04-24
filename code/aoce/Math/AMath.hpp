#pragma once

#include <assert.h>

#include <memory>
#include <vector>

#include "../Aoce.h"
#include "../Aoce.hpp"

namespace aoce {

struct vec3 {
    float x = 0;
    float y = 0;
    float z = 0;
    inline bool operator==(const vec3 &right) {
        return this->x == right.x && this->y == right.y && this->z == right.z;
    }
    inline float &operator[](int32_t idx) {
        assert(idx >= 0 && idx < 3);
        return *(&x + idx);
    }
    inline const float &operator[](int32_t idx) const {
        assert(idx >= 0 && idx < 3);
        return *(&x + idx);
    }
    inline vec3 operator*(const float &scale) {
        vec3 temp = {};
        temp.x = this->x * scale;
        temp.y = this->y * scale;
        temp.z = this->z * scale;
        return temp;
    }
};

struct vec4 {
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;
    vec4() {
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->x = 0;
    }
    vec4(float x, float y, float z, float w) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->x = w;
    }
    vec4(vec3 vec, float w) {
        this->x = vec.x;
        this->y = vec.y;
        this->z = vec.z;
        this->x = w;
    }
    inline bool operator==(const vec4 &right) {
        return this->x == right.x && this->y == right.y && this->z == right.z &&
               this->w == right.w;
    }
    inline float &operator[](int32_t idx) {
        assert(idx >= 0 && idx < 4);
        return *(&x + idx);
    }
    inline const float &operator[](int32_t idx) const {
        assert(idx >= 0 && idx < 4);
        return *(&x + idx);
    }
    inline vec4 operator*(const float &scale) {
        vec4 temp = {};
        temp.x = this->x * scale;
        temp.y = this->y * scale;
        temp.z = this->z * scale;
        temp.w = this->w * scale;
        return temp;
    }
};

struct Mat4x4 {
    vec4 col0 = {1.0, 0.0, 0.0, 0.0};
    vec4 col1 = {0.0, 1.0, 0.0, 0.0};
    vec4 col2 = {0.0, 0.0, 1.0, 0.0};
    vec4 col3 = {0.0, 0.0, 0.0, 1.0};
    inline bool operator==(const Mat4x4 &right) {
        return this->col0 == right.col0 && this->col1 == right.col1 &&
               this->col2 == right.col2 && this->col3 == right.col3;
    }
    inline vec4 &operator[](int32_t idx) {
        assert(idx >= 0 && idx < 4);
        return *(&col0 + idx);
    }
    inline const vec4 &operator[](int32_t idx) const {
        assert(idx >= 0 && idx < 4);
        return *(&col0 + idx);
    }
};

ACOE_EXPORT void identMat(Mat4x4 &mat);

ACOE_EXPORT Mat4x4 matMult(const Mat4x4 &a, const Mat4x4 &b);

ACOE_EXPORT vec3 transformMat(const Mat4x4 &mat, const vec3 &trans);

ACOE_EXPORT Mat4x4 scaleIdentMat(const vec3 &scale);

ACOE_EXPORT Mat4x4 scaleMat(const Mat4x4 &mat, const vec3 &scale);

ACOE_EXPORT Mat4x4 saturateMat(const Mat4x4 &mat, const float &saturate);

}  // namespace aoce