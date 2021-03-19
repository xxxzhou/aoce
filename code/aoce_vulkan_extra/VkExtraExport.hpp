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

enum class ConvertType { other = 0, rgba82rgba32f, rgba32f2rgba8 };

struct KernelSizeParamet {
    int32_t kernelSizeX = 5;
    int32_t kernelSizeY = 5;

    inline bool operator==(const KernelSizeParamet& right) {
        return this->kernelSizeX == right.kernelSizeX &&
               this->kernelSizeY == right.kernelSizeY;
    }
};

struct GaussianBlurParamet {
    int32_t blurRadius = 4;
    // sigma值越小,整个分布长度范围越大，原始值占比越高，周围占比越低
    float sigma = 2.0f;

    inline bool operator==(const GaussianBlurParamet& right) {
        return this->blurRadius == right.blurRadius &&
               this->sigma == right.sigma;
    }
};

struct ChromKeyParamet {
    // 0.2 控制亮度的强度系数
    float lumaMask = 2.0f;
    vec3 chromaColor;
    // 用环境光补受蓝绿幕影响的像素(简单理解扣像结果要放入的环境光的颜色)
    float ambientScale = 1.f;
    vec3 ambientColor;
    // 0.4
    float alphaCutoffMin = 0.1f;
    // 0.41
    float alphaCutoffMax = 0.2f;
    float alphaExponent = 1.f;
    // 0.8
    float despillCuttofMax = 0.8f;
    float despillExponent = 1.f;
};

struct AdaptiveThresholdParamet {
    int32_t boxSize = 10;
    float offset = 0.05f;
};

struct GuidedParamet {
    int32_t boxSize = 10;
    // //0.1-0.000001
    float eps = 0.00001f;
};

struct GuidedMattingParamet {
    GuidedParamet guided = {};
};

AOCE_VULKAN_EXTRA_EXPORT ITLayer<KernelSizeParamet>* createBoxFilterLayer(
    ImageType imageType = ImageType::rgba8);

AOCE_VULKAN_EXTRA_EXPORT ITLayer<GaussianBlurParamet>* createGaussianBlurLayer(
    ImageType imageType = ImageType::rgba8);

AOCE_VULKAN_EXTRA_EXPORT ITLayer<ChromKeyParamet>* createChromKeyLayer();

AOCE_VULKAN_EXTRA_EXPORT ITLayer<AdaptiveThresholdParamet>*
createAdaptiveThresholdLayer();

AOCE_VULKAN_EXTRA_EXPORT ITLayer<GuidedParamet>* createGuidedLayer();

AOCE_VULKAN_EXTRA_EXPORT ITLayer<ReSizeParamet>* createResizeLayer(
    ImageType imageType = ImageType::rgba8);

AOCE_VULKAN_EXTRA_EXPORT ITLayer<GuidedMattingParamet>*
createGuidedMattingLayer(BaseLayer* mattingLayer);

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createLuminanceLayer();

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createAlphaShowLayer();

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createConvertImageLayer();

}  // namespace vulkan
}  // namespace aoce
