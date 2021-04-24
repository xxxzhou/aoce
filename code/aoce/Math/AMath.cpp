#include "AMath.hpp"

#include "../Aoce.h"

namespace aoce {

#define RLUM (0.3f)
#define GLUM (0.59f)
#define BLUM (0.11f)

void identMat(Mat4x4& mat) {
    mat.col0 = {1.0, 0.0, 0.0, 0.0};
    mat.col1 = {0.0, 1.0, 0.0, 0.0};
    mat.col2 = {0.0, 0.0, 1.0, 0.0};
    mat.col3 = {0.0, 0.0, 0.0, 1.0};
}

Mat4x4 matMult(const Mat4x4& a, const Mat4x4& b) {
    Mat4x4 temp = {};
    for (int32_t y = 0; y < 4; y++) {
        for (int32_t x = 0; x < 4; x++) {
            temp[y][x] = b[y][0] * a[0][x] + b[y][1] * a[1][x] +
                         b[y][2] * a[2][x] + b[y][3] * a[3][x];
        }
    }
    return temp;
}

vec3 transformMat(const Mat4x4& mat, const vec3& trans) {
    vec3 temp = {};
    for (int32_t i = 0; i < 3; i++) {
        temp[i] = trans[0] * mat[0][i] + trans[1] * mat[1][i] +
                  trans[2] * mat[2][i] + mat[3][i];
    }
    return temp;
}

Mat4x4 scaleIdentMat(const vec3& scale) {
    Mat4x4 temp = {};
    identMat(temp);
    for (int32_t i = 0; i < 3; i++) {
        temp[i][i] = scale[i];
    }
    return temp;
}

Mat4x4 scaleMat(const Mat4x4& mat, const vec3& scale) {
    Mat4x4 temp = scaleIdentMat(scale);
    return matMult(temp, mat);
}

Mat4x4 saturateMat(const Mat4x4& mat, const float& saturate) {
    vec3 lum = {RLUM, GLUM, BLUM};
    float sat = 1.0 - saturate;
    vec4 vecSat = {sat, sat, sat, 0};
    Mat4x4 temp = {};
    temp.col0 = vecSat * lum.x;
    temp.col1 = vecSat * lum.y;
    temp.col2 = vecSat * lum.z;
    temp.col3 = {0.0f, 0.0f, 0.0f, 1.0f};
    for (int32_t i = 0; i < 3; i++) {
        temp[i][i] = temp[i][i] + saturate;
    }
    return matMult(temp, mat);
}

}  // namespace aoce