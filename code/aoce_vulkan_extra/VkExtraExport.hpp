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

enum class ReduceOperate {
    sum,
    min,
    max,
};

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
    // 如果为0,根据blurRadius自动计算
    float sigma = 0.0f;

    inline bool operator==(const GaussianBlurParamet& right) {
        return this->blurRadius == right.blurRadius &&
               this->sigma == right.sigma;
    }
};

struct ChromKeyParamet {
    // 比较差异,确定使用亮度与颜色比例,值需大于0,值越大,亮度所占比例越大
    float lumaMask = 1.0f;
    // 需要扣除的颜色
    vec3 chromaColor = {};
    // 用环境光补受蓝绿幕影响的像素(简单理解扣像结果要放入的环境光的颜色)
    float ambientScale = 0.f;
    // 环境光颜色
    vec3 ambientColor = {};
    // 比较差异相差的最少值(少于这值会放弃alpha)
    float alphaCutoffMin = 0.2f;
    // 比较后的alpha系数增亮
    float alphaScale = 10.0f;
    // 比较后的alpha指数增亮
    float alphaExponent = 0.1f;
    // 溢漏(蓝绿幕对物体的影响)系数,这部分颜色扣除并用环境补起
    float despillScale = 0.0f;
    // 溢漏(蓝绿幕对物体的影响)指数
    float despillExponent = 0.1f;
    inline bool operator==(const ChromKeyParamet& right) {
        return this->lumaMask == right.lumaMask &&
               this->chromaColor == right.chromaColor &&
               this->ambientScale == right.ambientScale &&
               this->ambientColor == right.ambientColor &&
               this->alphaCutoffMin == right.alphaCutoffMin &&
               this->alphaScale == right.alphaScale &&
               this->alphaExponent == right.alphaExponent &&
               this->despillScale == right.despillScale &&
               this->despillExponent == right.despillExponent;
    }
};

struct AdaptiveThresholdParamet {
    int32_t boxSize = 10;
    float offset = 0.05f;
};

struct GuidedParamet {
    int32_t boxSize = 10;
    // //0.1-0.0000001
    float eps = 0.000001f;
};

struct GuidedMattingParamet {
    GuidedParamet guided = {};
};

struct HarrisCornerDetectionParamet {
    float edgeStrength = 1.0f;
    GaussianBlurParamet blueParamet = {4, 0.0f};
    float harris = 0.04f;
    float sensitivity = 5.0f;
    float threshold = 0.2f;
};

struct BilateralParamet {
    // 模糊周边的半径(圆形)
    int32_t kernelSize = 5;
    // 同高斯模糊的sigma,值越小,周边占比越小
    float sigma_spatial = 10.0f;
    // sigma_spatial是距离间系数,sigma_color是颜色差异比较
    // 同上,这值越小,颜色差异大的部分占比小
    float sigma_color = 10.0f;

    inline bool operator==(const BilateralParamet& right) {
        return this->kernelSize == right.kernelSize &&
               this->sigma_spatial == right.sigma_spatial &&
               this->sigma_color == right.sigma_color;
    }
};

struct BulgeDistortionParamet {
    float aspectRatio = 1920.0f / 1080.0f;
    float centerX = 0.5f;
    float centerY = 0.5f;
    float radius = 0.25f;
    float scale = 0.5f;

    inline bool operator==(const BulgeDistortionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->centerX == right.centerX &&
               this->centerY == right.centerY && this->radius == right.radius &&
               this->scale == right.scale;
    }
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

AOCE_VULKAN_EXTRA_EXPORT ITLayer<HarrisCornerDetectionParamet>*
createHarrisCornerDetectionLayer();

AOCE_VULKAN_EXTRA_EXPORT ITLayer<float>* createAverageLuminanceThresholdLayer();

AOCE_VULKAN_EXTRA_EXPORT ITLayer<BilateralParamet>* createBilateralLayer();

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createLuminanceLayer();

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createAlphaShowLayer();

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createAlphaShow2Layer();

AOCE_VULKAN_EXTRA_EXPORT BaseLayer* createConvertImageLayer();

}  // namespace vulkan
}  // namespace aoce
