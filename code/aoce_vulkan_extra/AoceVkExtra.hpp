#pragma once

#include "Aoce/Aoce.hpp"
#include "Aoce/Layer/BaseLayer.hpp"
#include "Aoce/Math/AMath.hpp"

#ifdef _WIN32
#if defined(AOCE_VULKAN_EXTRA_EXPORT_DEFINE)
#define AOCE_VE_EXPORT __declspec(dllexport)
#else
#define AOCE_VE_EXPORT __declspec(dllimport)
#endif
#else
#define AOCE_VE_EXPORT
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

struct HarrisDetectionBaseParamet {
    float edgeStrength = 1.0f;
    GaussianBlurParamet blueParamet = {4, 0.0f};
    float threshold = 0.2f;

    inline bool operator==(const HarrisDetectionBaseParamet& right) {
        return this->edgeStrength == right.edgeStrength &&
               this->blueParamet == right.blueParamet &&
               this->threshold == right.threshold;
    }
};

struct HarrisCornerDetectionParamet {
    HarrisDetectionBaseParamet harrisBase = {};
    float harris = 0.04f;
    float sensitivity = 5.0f;
};

struct NobleCornerDetectionParamet {
    HarrisDetectionBaseParamet harrisBase = {};
    float sensitivity = 5.0f;
};

struct CannyEdgeDetectionParamet {
    GaussianBlurParamet blueParamet = {4, 0.0f};
    float minThreshold = 0.1f;
    float maxThreshold = 0.4f;
};

struct FASTFeatureParamet {
    int32_t boxSize = 5;
    float offset = 1.0f;
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

struct DistortionParamet {
    float aspectRatio = 1.0f;
    vec2 center = {0.5f, 0.5f};
    float radius = 0.25f;
    float scale = 0.5f;

    inline bool operator==(const DistortionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius &&
               this->scale == right.scale;
    }
};

struct PositionParamet {
    float aspectRatio = 1.0f;
    vec2 center = {0.5f, 0.5f};
    float radius = 0.25f;

    inline bool operator==(const PositionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius;
    }
};

struct SelectiveParamet {
    float aspectRatio = 1.0f;
    vec2 center = {0.5f, 0.5f};
    float radius = 0.25f;
    float size = 0.125f;
    inline bool operator==(const SelectiveParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius &&
               this->size == size;
    }
};

struct SphereRefractionParamet {
    float aspectRatio = 1.0f;
    vec2 center = {0.5f, 0.5f};
    float radius = 0.25f;
    float refractiveIndex = 0.71f;
    inline bool operator==(const SphereRefractionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius &&
               this->refractiveIndex == refractiveIndex;
    }
};

struct BulrPositionParamet {
    GaussianBlurParamet gaussian = {};
    PositionParamet bulrPosition = {};
};

// 马赛克
struct PixellateParamet {
    float fractionalWidthOfPixel = 0.01f;
    float aspectRatio = 1.0f;
    inline bool operator==(const PixellateParamet& right) {
        return this->fractionalWidthOfPixel == right.fractionalWidthOfPixel &&
               this->aspectRatio == right.aspectRatio;
    }
};

struct BlurSelectiveParamet {
    GaussianBlurParamet gaussian = {};
    SelectiveParamet bulrPosition = {};
};

struct ColorMatrixParamet {
    float intensity = 1.0f;
    Mat4x4 mat = {};

    inline bool operator==(const ColorMatrixParamet& right) {
        return this->intensity == right.intensity && this->mat == right.mat;
    }
};

struct CropParamet {
    float centerX = 0.5f;
    float centerY = 0.5f;
    float width = 0.5f;
    float height = 0.5f;

    inline bool operator==(const CropParamet& right) {
        return this->centerX == right.centerX &&
               this->centerY == right.centerY && this->width == right.width &&
               this->height == right.height;
    }
};

struct CrosshatchParamet {
    float crossHatchSpacing = 0.03f;
    float lineWidth = 0.003f;
    inline bool operator==(const CrosshatchParamet& right) {
        return this->crossHatchSpacing == right.crossHatchSpacing &&
               this->lineWidth == right.lineWidth;
    }
};

struct FalseColorParamet {
    vec3 firstColor = {0.0f, 0.0f, 0.5f};
    vec3 secondColor = {1.0f, 0.0f, 0.0f};
    inline bool operator==(const FalseColorParamet& right) {
        return this->firstColor == right.firstColor &&
               this->secondColor == right.secondColor;
    }
};

// 去雾
struct HazeParamet {
    // Strength of the color applied. Values between -.3 and .3 are best
    float distance = 0.0f;
    // Amount of color change. Values between -.3 and .3 are best
    float slope = 0.0f;
    inline bool operator==(const HazeParamet& right) {
        return this->distance == right.distance && this->slope == right.slope;
    }
};

// 调整图像的阴影和高光
struct HighlightShadowParamet {
    // 0 - 1, increase to lighten shadows.
    float shadows = 0.0f;
    // 0 - 1, decrease to darken highlights.
    float highlights = 1.0f;
    inline bool operator==(const HighlightShadowParamet& right) {
        return this->shadows == right.shadows &&
               this->highlights == right.highlights;
    }
};

// 允许您使用颜色和强度独立地着色图像的阴影和高光
struct HighlightShadowTintParamet {
    float shadowTintIntensity = 0.0f;
    vec3 shadowTintColor = {1.0f, 0.0f, 0.0f};
    float highlightTintIntensity = 1.0f;
    vec3 highlightTintColor = {0.0f, 0.0f, 1.0f};
    inline bool operator==(const HighlightShadowTintParamet& right) {
        return this->shadowTintIntensity == right.shadowTintIntensity &&
               this->shadowTintColor == right.shadowTintColor &&
               this->highlightTintIntensity == right.highlightTintIntensity &&
               this->highlightTintColor == right.highlightTintColor;
    }
};

struct IOSBlurParamet {
    float sacle = 4.0f;
    GaussianBlurParamet blurParamet = {12, 0.0f};
    float saturation = 0.8f;
    float range = 0.6f;
};

struct LevelsParamet {
    vec3 minVec = {0.0, 0.0, 0.0};
    vec3 gammaVec = {1.0, 1.0, 1.0};
    vec3 maxVec = {1.0, 1.0, 1.0};
    vec3 minOut = {0.0, 0.0, 0.0};
    vec3 maxOunt = {1.0, 1.0, 1.0};
};

struct MonochromeParamet {
    float intensity = 1.0f;
    vec3 color = {0.6f, 0.45f, 0.3f};
    inline bool operator==(const MonochromeParamet& right) {
        return this->intensity == right.intensity && this->color == right.color;
    }
};

struct MotionBlurParamet {
    float blurSize = 1.0f;
    float blurAngle = 0.0f;
    inline bool operator==(const MotionBlurParamet& right) {
        return this->blurSize == right.blurSize &&
               this->blurAngle == right.blurAngle;
    }
};

struct PoissonParamet {
    float percent = 0.5f;
    int32_t iterationNum = 10;
};

struct PerlinNoiseParamet {
    float scale = 8.0f;
    vec4 colorStart = {0.0, 0.0, 0.0, 1.0};
    vec4 colorFinish = {1.0, 1.0, 1.0, 1.0};
    inline bool operator==(const PerlinNoiseParamet& right) {
        return this->colorStart == right.colorStart &&
               this->colorFinish == right.colorFinish &&
               this->scale == right.scale;
    }
};

struct PolarPixellateParamet {
    vec2 center = {0.5f, 0.5f};
    vec2 size = {0.05, 0.05};

    inline bool operator==(const PolarPixellateParamet& right) {
        return this->center == right.center && this->size == right.size;
    }
};

struct PolkaDotParamet {
    float dotScaling = 0.90f;
    float fractionalWidthOfPixel = 0.01f;
    float aspectRatio = 1.0f;
    inline bool operator==(const PolkaDotParamet& right) {
        return this->dotScaling == right.dotScaling &&
               this->fractionalWidthOfPixel == right.fractionalWidthOfPixel &&
               this->aspectRatio == right.aspectRatio;
    }
};

// 图像锐化
struct SharpenParamet {
    int offset = 1;
    // [-4,4]
    float sharpness = 0.0f;
    inline bool operator==(const SharpenParamet& right) {
        return this->offset == right.offset &&
               this->sharpness == right.sharpness;
    }
};

struct SkinToneParamet {
    // [-1,1]
    float skinToneAdjust = 0.0f;
    float skinHue = 0.05f;
    float skinHueThreshold = 40.0f;
    float maxHueShift = 0.25f;
    float maxSaturationShift = 0.4f;
    // [0,1]
    int upperSkinToneColor = 0;

    inline bool operator==(const SkinToneParamet& right) {
        return this->skinToneAdjust == right.skinToneAdjust &&
               this->skinHue == right.skinHue &&
               this->skinHueThreshold == right.skinHueThreshold &&
               this->upperSkinToneColor == right.upperSkinToneColor &&
               this->maxSaturationShift == right.maxSaturationShift &&
               this->maxHueShift == right.maxHueShift;
    }
};

struct ToonParamet {
    float threshold = 0.2f;
    float quantizationLevels = 10.0f;
    inline bool operator==(const ToonParamet& right) {
        return this->threshold == right.threshold &&
               this->quantizationLevels == right.quantizationLevels;
    }
};

class LookupLayer : public ILayer {
   public:
    virtual void loadLookUp(uint8_t* data, int32_t size) = 0;
};

class HSBLayer : public ILayer {
   public:
    virtual void reset() = 0;
    virtual void rotateHue(const float& h) = 0;
    virtual void adjustSaturation(const float& h) = 0;
    virtual void adjustBrightness(const float& h) = 0;
};

typedef std::function<void(vec4 motion)> motionHandle;

class MotionDetectorLayer : public ITLayer<float> {
   protected:
    motionHandle onMotionEvent;

   public:
    MotionDetectorLayer(){};
    virtual ~MotionDetectorLayer(){};

   public:
    inline void setMotionHandle(motionHandle handle) { onMotionEvent = handle; }
};

class PerlinNoiseLayer : public ITLayer<PerlinNoiseParamet> {
   public:
    PerlinNoiseLayer(){};
    virtual ~PerlinNoiseLayer(){};

   public:
    virtual void setImageSize(int32_t width, int32_t height) = 0;
};
}  // namespace vulkan
}  // namespace aoce