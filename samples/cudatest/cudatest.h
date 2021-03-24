#pragma once

#include <cuda.h>
#include <cuda/cuda_common.h>
#include <cuda/helper_math.h>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "aoce/Aoce.hpp"

using namespace cv;
using namespace cv::cuda;

struct Keying {
    // 0.2 控制亮度的强度系数
    float lumaMask = 2.0f;
    float3 chromaColor;
    // 用环境光补受蓝绿幕影响的像素(简单理解扣像结果要放入的环境光的颜色)
    float ambientScale = 1.f;
    float3 ambientColor;
    // 0.4
    float alphaCutoffMin = 0.1f;
    // 0.41
    float alphaScale = 0.2f;
    float alphaExponent = 1.f;
    // 0.8
    float despillScale = 0.8f;
    float despillExponent = 1.f;
};

inline __host__ __device__ float3 extractColor(float3 color, float lumaMask) {
    float luma = dot(color, make_float3(1.f));
    float colorMask =
        exp(-luma * 2 * 3.1415926 / lumaMask);  // exp(-~)=0,exp(0)=1
    // luma越小/lumaMask越大,colorMask越大,颜色越靠近luma
    float3 dcolor = lerp(color, make_float3(luma), colorMask);
    dcolor = dcolor / (dot(dcolor, make_float3(2.f)));
    return dcolor;
}

inline __global__ void combin(PtrStepSz<uchar3> source, PtrStepSz<uchar> gray,
                              PtrStepSz<float4> dest) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < source.cols && idy < source.rows) {
        dest(idy, idx) = rgbauchar42float4(
            make_uchar4(source(idy, idx).x, source(idy, idx).y,
                        source(idy, idx).z, gray(idy, idx)));
    }
}

//采用UE4里的chroma-key 颜色与导向图分开成像
// https://shaderbits.com/blog/green-screen-live-in-ue4
inline __global__ void simpleKeyingUE4(PtrStepSz<uchar4> source,
                                       PtrStepSz<float4> dest, Keying ubo) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    // const int id = threadIdx.x + threadIdx.y*blockDim.x;
    // idx与idy可能大于对应长度与宽度,这样程序会出错
    if (idx < source.cols && idy < source.rows) {
        // https://www.unrealengine.com/zh-CN/tech-blog/setting-up-a-chroma-key-material-in-ue4
        float4 color = rgbauchar42float4(source(idy, idx));
        float3 inputColor = make_float3(color);
        float3 chromaColor = ubo.chromaColor;
        float3 ambientColor = ubo.ambientColor;
        float3 color1 = extractColor(chromaColor, ubo.lumaMask);
        float3 color2 = extractColor(inputColor, ubo.lumaMask);
        float3 subColor = color1 - color2;
        float diffSize = length(subColor);
        float minClamp =
            max((diffSize - ubo.alphaCutoffMin) * ubo.alphaScale, 0.0);
        // 扣像alpha
        float alpha = clamp(pow(minClamp, ubo.alphaExponent), 0.0, 1.0);
        // 受扣像背景影响的颜色alpha
        float despillAlpha =
            1.0f - clamp(pow(minClamp, ubo.despillExponent), 0.0, 1.0);
        // 亮度系数
        float3 lumaFactor = make_float3(0.3f, 0.59f, 0.11f);
        float3 dcolor = inputColor;
        // 去除扣像背景影响的颜色
        dcolor -= inputColor * chromaColor * despillAlpha * ubo.despillScale;
        // 添加环境光收益
        dcolor += inputColor * lumaFactor * ambientColor * ubo.ambientScale *
                  despillAlpha;
        dest(idy, idx) = make_float4(inputColor, alpha);
    }
}

inline __global__ void findMatrix(PtrStepSz<float4> source,
                                  PtrStepSz<float3> dest,
                                  PtrStepSz<float3> dest1,
                                  PtrStepSz<float3> dest2) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < source.cols && idy < source.rows) {
        float4 scolor =
            source(idy, idx);  // rgbauchar42float4(source(idy, idx));
        float3 color = make_float3(scolor);

        dest(idy, idx) = color * scolor.w;
        dest1(idy, idx) = color.x * color;
        dest2(idy, idx) = make_float3(color.y * color.y, color.y * color.z,
                                      color.z * color.z);
    }
}

//导向滤波求值 Guided filter
//论文地址http://kaiminghe.com/publications/pami12guidedfilter.pdf
// https://blog.csdn.net/baimafujinji/article/details/74750283
inline __global__ void guidedFilter(PtrStepSz<float4> source,
                                    PtrStepSz<float3> col1,
                                    PtrStepSz<float3> col2,
                                    PtrStepSz<float3> col3,
                                    PtrStepSz<float4> dest, float eps) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < source.cols && idy < source.rows) {
        float4 color = source(idy, idx);
        float3 mean_I = make_float3(color);
        float mean_p = color.w;
        float3 mean_Ip = col1(idy, idx);  // rgbauchar32float3(col1(idy, idx));
        float3 var_I_r = col2(idy, idx) - mean_I.x * mean_I;
        float3 var_I_gbxfv = col3(idy, idx);
        float gg = var_I_gbxfv.x - mean_I.y * mean_I.y;
        float gb = var_I_gbxfv.y - mean_I.y * mean_I.z;
        float bb = var_I_gbxfv.z - mean_I.z * mean_I.z;

        float3 cov_Ip = mean_Ip - mean_I * mean_p;
        float3 col0 = var_I_r + make_float3(eps, 0.f, 0.f);
        float3 col1 = make_float3(var_I_r.y, gg + eps, gb);
        float3 col2 = make_float3(var_I_r.z, gb, bb + eps);

        float3 invCol0 = make_float3(1.f, 0.f, 0.f);
        float3 invCol1 = make_float3(0.f, 1.f, 0.f);
        float3 invCol2 = make_float3(0.f, 0.f, 1.f);
        inverseMat3x3(col0, col1, col2, invCol0, invCol1, invCol2);
        // ax+b,a,b是其线性关系解
        float3 a = mulMat(cov_Ip, invCol0, invCol1, invCol2);
        float b = mean_p - dot(a, mean_I);

        dest(idy, idx) = make_float4(a, b);
    }
}

inline __global__ void guidedFilterResult(PtrStepSz<float4> source,
                                          PtrStepSz<float4> guid,
                                          PtrStepSz<uchar4> dest,
                                          PtrStepSz<uchar> destp) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < source.cols && idy < source.rows) {
        float4 color =
            source(idy, idx);  // rgbauchar42float4(source(idy, idx));//I
        float4 mean = guid(idy, idx);
        float alpha = clamp(
            color.x * mean.x + color.y * mean.y + color.z * mean.z + mean.w,
            0.f, 1.f);
        float3 rgb = make_float3(color);
        dest(idy, idx) = rgbafloat42uchar4(make_float4(rgb, alpha));
        destp(idy, idx) = (uchar)(__saturatef(alpha) * 255.0f);
    }
}