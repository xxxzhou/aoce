#include "WrapMat.hpp"

namespace aoce {

// 最小二乘法 https://www.it610.com/article/1278554388192837632.htm

WrapMat getAffineTransform(const vec2 src[], const vec2 dst[]) {
    WrapMat mat = {};
    double a[36] = {};
    double b[6] = {0};

    // for (int i = 0; i < 3; i++) {
    //     int j = i * 12;
    //     int k = i * 12 + 6;
    //     a[j] = a[k + 3] = src1[i].x;
    //     a[j + 1] = a[k + 4] = src1[i].y;
    //     a[j + 2] = a[k + 5] = 1;
    //     a[j + 3] = a[j + 4] = a[j + 5] = 0;
    //     a[k] = a[k + 1] = a[k + 2] = 0;
    //     b[i * 2] = src2[i].x;
    //     b[i * 2 + 1] = src2[i].y;
    // }

    return mat;
}

}  // namespace aoce