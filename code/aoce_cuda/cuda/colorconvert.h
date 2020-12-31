#pragma once
#include <Aoce.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_functions.h>

#include "CudaTypes.hpp"
#include "cuda_common.h"

// 这几个文件只用于nvcc编译,不会污染别的头文件
using namespace aoce;
using namespace aoce::cuda;

//长宽用目标的(20.05.29 记录下后续改进YUV420P/SP,长度全一半处理,现在读重复)
template <int32_t yuvpType>
__global__ void yuv2rgb(PtrStepSz<uchar> source, PtrStepSz<uchar4> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (yuvpType == 1 || yuvpType == 2) {
        if (idx < dest.width && idy < dest.height) {
            uchar y = source(idy, idx);
            uchar u = 0;
            uchar v = 0;
            float3 yuv = make_float3(0.f, 0.f, 0.f);
            //编译时优化掉选择
            if (yuvpType == 1) {
                int halfidx = idx >> 1;
                int halfidy = idy >> 1;
                u = source(halfidy + dest.height, halfidx * 2);
                v = source(halfidy + dest.height, halfidx * 2 + 1);
            }
            if (yuvpType == 6) {
                int2 nuv =
                    u12u2(u22u1(make_int2(idx / 2, idy / 2), dest.width / 2),
                          dest.width);
                u = source(nuv.y + dest.height, nuv.x);
                v = source(nuv.y + dest.height * 5 / 4, nuv.x);
            }
            yuv = rgbauchar32float3(make_uchar3(y, u, v));
            dest(idy, idx) = rgbafloat42uchar4(make_float4(yuv2Rgb(yuv), 1.f));
        }
    }
    if (yuvpType == 3) {
        if (idx < dest.width / 2 && idy < dest.height) {
            uchar y1 = source(idy, idx * 2);
            uchar y2 = source(idy, idx * 2 + 1);
            int2 nuv =
                u12u2(u22u1(make_int2(idx, idy), dest.width / 2), dest.width);
            uchar u = source(nuv.y + dest.height, nuv.x);
            uchar v = source(nuv.y + dest.height * 3 / 2, nuv.x);
            float3 yuv1 = rgbauchar32float3(make_uchar3(y1, u, v));
            float3 yuv2 = rgbauchar32float3(make_uchar3(y2, u, v));
            dest(idy, idx * 2) =
                rgbafloat42uchar4(make_float4(yuv2Rgb(yuv1), 1.f));
            dest(idy, idx * 2 + 1) =
                rgbafloat42uchar4(make_float4(yuv2Rgb(yuv2), 1.f));
        }
    }
}

//长宽用源的packed yuyv bitx/yoffset 0/0,yvyu 2/0,uyvy 0/1 source 一点分二点
inline __global__ void yuv2rgb(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                               int bitx, int yoffset) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < source.width && idy < source.height) {
        uchar4 yuyv = source(idy, idx);
        uchar ayuyv[4]{yuyv.x, yuyv.y, yuyv.z, yuyv.w};
        uchar y1 = ayuyv[yoffset];
        uchar u = ayuyv[bitx + (1 - yoffset)];
        uchar y2 = ayuyv[yoffset + 2];
        uchar v = ayuyv[(2 - bitx) + (1 - yoffset)];
        float3 yuv = rgbauchar32float3(make_uchar3(y1, u, v));
        dest(idy, idx * 2) = rgbafloat42uchar4(make_float4(yuv2Rgb(yuv), 1.f));
        yuv = rgbauchar32float3(make_uchar3(y2, u, v));
        dest(idy, idx * 2 + 1) =
            rgbafloat42uchar4(make_float4(yuv2Rgb(yuv), 1.f));
    }
}

template <int32_t yuvpType>
__global__ void rgb2yuv(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (yuvpType == 3) {
        if (idx < source.width / 2 && idy < source.height) {
            int2 uvt = make_int2(idx * 2, idy);
            int2 uvb = make_int2(idx * 2 + 1, idy);
            //由一半的长方体扩展到全部的长文体
            int2 nuv = u12u2(u22u1(make_int2(idx, idy), source.width / 2),
                             source.width);
            float3 yuvt =
                rgb2Yuv(make_float3(rgbauchar42float4(source(uvt.y, uvt.x))));
            float3 yuvb =
                rgb2Yuv(make_float3(rgbauchar42float4(source(uvb.y, uvb.x))));
            //左边的Y
            dest(uvt.y, uvt.x) = rgbafloat2ucha1(yuvt.x);
            //右边的Y
            dest(uvb.y, uvb.x) = rgbafloat2ucha1(yuvb.x);
            // U
            dest(source.height + nuv.y, nuv.x) =
                rgbafloat2ucha1((yuvt.y + yuvb.y) / 2.0f);
            // V
            dest(source.height * 3 / 2 + nuv.y, nuv.x) =
                rgbafloat2ucha1((yuvt.z + yuvb.z) / 2.0f);
        }
    }
    if (yuvpType == 1 || yuvpType == 2) {
        if (idx < source.width / 2 && idy < source.height / 2) {
            int2 uvlt = make_int2(idx * 2, idy * 2);
            int2 uvlb = make_int2(idx * 2, idy * 2 + 1);
            int2 uvrt = make_int2(idx * 2 + 1, idy * 2);
            int2 uvrb = make_int2(idx * 2 + 1, idy * 2 + 1);
            float3 yuvlt =
                rgb2Yuv(make_float3(rgbauchar42float4(source(uvlt.y, uvlt.x))));
            float3 yuvlb =
                rgb2Yuv(make_float3(rgbauchar42float4(source(uvlb.y, uvlb.x))));
            float3 yuvrt =
                rgb2Yuv(make_float3(rgbauchar42float4(source(uvrt.y, uvrt.x))));
            float3 yuvrb =
                rgb2Yuv(make_float3(rgbauchar42float4(source(uvrb.y, uvrb.x))));
            float3 ayuv = yuvlt + yuvlb + yuvrt + yuvrb;
            if (yuvpType == 1) {
                int2 uindex = make_int2(idx * 2, source.height + idy);
                int2 vindex = make_int2(idx * 2 + 1, source.height + idy);
                dest(uindex.y, uindex.x) = rgbafloat2ucha1(ayuv.y / 4.0f);
                dest(vindex.y, vindex.x) = rgbafloat2ucha1(ayuv.z / 4.0f);
            }
            if (yuvpType == 2) {
                int2 nuv = u12u2(u22u1(make_int2(idx, idy), source.width / 2),
                                 source.width);
                dest(source.height + nuv.y, nuv.x) =
                    rgbafloat2ucha1(ayuv.y / 4.0f);
                dest(source.height * 5 / 4 + nuv.y, nuv.x) =
                    rgbafloat2ucha1(ayuv.z / 4.0f);
            }
            dest(uvlt.y, uvlt.x) = rgbafloat2ucha1(yuvlt.x);
            dest(uvlb.y, uvlb.x) = rgbafloat2ucha1(yuvlb.x);
            dest(uvrt.y, uvrt.x) = rgbafloat2ucha1(yuvrt.x);
            dest(uvrb.y, uvrb.x) = rgbafloat2ucha1(yuvrb.x);
        }
    }
}

// dest 一点分二点
inline __global__ void rgb2yuv(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                               int bitx, int yoffset) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < dest.width && idy < dest.height) {
        float4 rgba1 = rgbauchar42float4(source(idy, idx * 2));
        float4 rgba2 = rgbauchar42float4(source(idy, idx * 2 + 1));
        float3 yuv1 = rgb2Yuv(make_float3(rgba1));
        float3 yuv2 = rgb2Yuv(make_float3(rgba2));
        float4 ryuyv = make_float4(yuv1.x, (yuv1.y + yuv2.y) / 2.f, yuv2.x,
                                   (yuv1.z + yuv2.z) / 2.f);
        float yuyv[4] = {ryuyv.x, ryuyv.y, ryuyv.z, ryuyv.w};
        float4 syuyv =
            make_float4(yuyv[yoffset], yuyv[bitx + (1 - yoffset)],
                        yuyv[yoffset + 2], yuyv[(2 - bitx) + (1 - yoffset)]);
        dest(idy, idx) = rgbafloat42uchar4(syuyv);
    }
}

inline __global__ void rgb2rgba(PtrStepSz<uchar3> source,
                                PtrStepSz<uchar4> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < source.width && idy < source.height) {
        uchar3 rgb = source(idy, idx);
        dest(idy, idx) = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
    }
}

inline __global__ void rgba2bgr(PtrStepSz<uchar4> source,
                                PtrStepSz<uchar3> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < source.width && idy < source.height) {
        uchar4 rgba = source(idy, idx);
        dest(idy, idx) = make_uchar3(rgba.z, rgba.y, rgba.x);
    }
}

inline __global__ void argb2rgba(PtrStepSz<uchar4> source,
                                 PtrStepSz<uchar4> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < source.width && idy < source.height) {
        uchar4 rgba = source(idy, idx);
        dest(idy, idx) = make_uchar4(rgba.y, rgba.z, rgba.w, rgba.x);
    }
}

inline __global__ void textureMap(PtrStepSz<uchar4> source,
                                  PtrStepSz<uchar4> dest,
                                  MapChannelParamet paramt) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < dest.width && idy < dest.height) {
        uchar4 rgba = source(idy, idx);
        uchar color[4] = {rgba.x, rgba.y, rgba.z, rgba.w};
        dest(idy, idx) = make_uchar4(color[paramt.red], color[paramt.green],
                                     color[paramt.blue], color[paramt.alpha]);
    }
}

inline __global__ void blend(PtrStepSz<uchar4> source,
                             PtrStepSz<uchar4> blendTex, PtrStepSz<uchar4> dest,
                             int32_t left, int32_t top, float opacity) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < dest.width && idy < dest.height) {
        float4 rgba = rgbauchar42float4(source(idy, idx));
        if (idx >= left && idx < left + blendTex.width && idy >= top &&
            idy < top + blendTex.height) {
            float4 rgba2 = rgbauchar42float4(blendTex(idy - top, idx - left));
            rgba = rgba2 * (1.f - opacity) + rgba * opacity;
        }
        dest(idy, idx) = rgbafloat42uchar4(rgba);
    }
}

inline __global__ void operate(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                               OperateParamet paramt) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < dest.width && idy < dest.height) {
        int ix = idx;
        int iy = idy;
        if (paramt.bFlipX) {
            ix = source.width - idx;
        }
        if (paramt.bFlipY) {
            iy = source.height - idy;
        }
        float4 rgba = rgbauchar42float4(source(iy, ix));
        float4 grgba =
            make_float4(powf(rgba.x, paramt.gamma), powf(rgba.y, paramt.gamma),
                        powf(rgba.z, paramt.gamma), rgba.w);
        dest(idy, idx) = rgbafloat42uchar4(grgba);
    }
}

inline __global__ void uchar2float(PtrStepSz<uchar4> source,
                                   PtrStepSz<float4> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < dest.width && idy < dest.height) {
        dest(idy, idx) = rgbauchar42float4(source(idy, idx));
    }
}
