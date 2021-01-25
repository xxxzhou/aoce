#pragma once

#include "Aoce/Aoce.hpp"
#include "Aoce/Layer/BaseLayer.hpp"

#ifdef _WIN32
#if defined(AOCE_VULKAN_EXTRA_EXPORT_DEFINE)
#define AOCE_VULKAN_EXTRA_EXPORT __declspec(dllexport)
#else
#define AOCE_VULKAN_EXTRA_EXPORT __declspec(dllimport)
#endif
#else
#define AOCE_VULKAN_EXTRA_EXPORT
#endif

namespace aoce {
namespace vulkan {

struct FilterParamet {
    int32_t kernelSizeX = 3;
    int32_t kernelSizeY = 3;
};

struct ChromKeyParamet {
    // 0.2 控制亮度的强度系数
    float lumaMask = 0.2f;
    vec3 chromaColor;
    // 用环境光补受蓝绿幕影响的像素(简单理解扣像结果要放入的环境光的颜色)
    float ambientScale = 1.f;
    vec3 ambientColor;
    // 0.4
    float alphaCutoffMin = 0.4f;
    // 0.41
    float alphaCutoffMax = 0.5f;
    float alphaExponent = 1.f;
    // 0.8
    float despillCuttofMax = 0.8f;
    float despillExponent = 1.f;
};

AOCE_VULKAN_EXTRA_EXPORT ITLayer<FilterParamet>* createBoxFilterLayer();

AOCE_VULKAN_EXTRA_EXPORT ITLayer<ChromKeyParamet>* createChromKeyLayer();

}  // namespace vulkan
}  // namespace aoce
