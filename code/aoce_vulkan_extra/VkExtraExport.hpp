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

struct HarrisCornerDetectionParamet {
    float edgeStrength = 1.0f;
    GaussianBlurParamet blueParamet = {4, 0.0f};
    float harris = 0.04f;
    float sensitivity = 5.0f;
    float threshold = 0.2f;
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

struct BulgeDistortionParamet {
    float aspectRatio = 1.0f;
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

struct BulrPosition {
    float aspectRatio = 1.0f;
    float centerX = 0.5f;
    float centerY = 0.5f;
    float radius = 0.25f;

    inline bool operator==(const BulrPosition& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->centerX == right.centerX &&
               this->centerY == right.centerY && this->radius == right.radius;
    }
};

struct BulrPositionParamet {
    GaussianBlurParamet gaussian = {};
    BulrPosition bulrPosition = {};
};

struct BlurSelective {
    float aspectRatio = 1.0f;
    float centerX = 0.5f;
    float centerY = 0.5f;
    float radius = 0.25f;
    float size = 0.125f;
    inline bool operator==(const BlurSelective& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->centerX == right.centerX &&
               this->centerY == right.centerY && this->radius == right.radius &&
               this->size == size;
    }
};

struct SphereRefractionParamet {
    float aspectRatio = 1.0f;
    float centerX = 0.5f;
    float centerY = 0.5f;
    float radius = 0.25f;
    float refractiveIndex = 0.71f;
    inline bool operator==(const SphereRefractionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->centerX == right.centerX &&
               this->centerY == right.centerY && this->radius == right.radius &&
               this->refractiveIndex == refractiveIndex;
    }
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
    BlurSelective bulrPosition = {};
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

AOCE_VE_EXPORT ITLayer<KernelSizeParamet>* createBoxFilterLayer(
    ImageType imageType = ImageType::rgba8);

AOCE_VE_EXPORT ITLayer<GaussianBlurParamet>* createGaussianBlurLayer(
    ImageType imageType = ImageType::rgba8);

AOCE_VE_EXPORT ITLayer<ChromKeyParamet>* createChromKeyLayer();

AOCE_VE_EXPORT ITLayer<AdaptiveThresholdParamet>*
createAdaptiveThresholdLayer();

AOCE_VE_EXPORT ITLayer<GuidedParamet>* createGuidedLayer();

AOCE_VE_EXPORT ITLayer<ReSizeParamet>* createResizeLayer(
    ImageType imageType = ImageType::rgba8);

AOCE_VE_EXPORT ITLayer<HarrisCornerDetectionParamet>*
createHarrisCornerDetectionLayer();

AOCE_VE_EXPORT ITLayer<float>* createAverageLuminanceThresholdLayer();

AOCE_VE_EXPORT ITLayer<BilateralParamet>* createBilateralLayer();

AOCE_VE_EXPORT BaseLayer* createAddBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createAlphaBlendLayer();

// 二输入,第一输入原始图像,第二输入512*512的lookup图
AOCE_VE_EXPORT LookupLayer* createLookupLayer();

AOCE_VE_EXPORT ITLayer<float>* createBrightnessLayer();

AOCE_VE_EXPORT ITLayer<BulgeDistortionParamet>* createBulgeDistortionLayer();

AOCE_VE_EXPORT ITLayer<CannyEdgeDetectionParamet>*
createCannyEdgeDetectionLayer();

AOCE_VE_EXPORT BaseLayer* createCGAColorspaceLayer();

AOCE_VE_EXPORT ITLayer<int32_t>* createDilationLayer();

AOCE_VE_EXPORT ITLayer<int32_t>* createErosionLayer();

AOCE_VE_EXPORT ITLayer<int32_t>* createClosingLayer();

AOCE_VE_EXPORT ITLayer<int32_t>* createOpeningLayer();

AOCE_VE_EXPORT BaseLayer* createColorBlendLayer();

AOCE_VE_EXPORT BaseLayer* createColorBurnBlendLayer();

AOCE_VE_EXPORT BaseLayer* createColorDodgeBlendLayer();

AOCE_VE_EXPORT BaseLayer* createColorInvertLayer();

AOCE_VE_EXPORT BaseLayer* createColorLBPLayer();

AOCE_VE_EXPORT ITLayer<float>* createContrastLayer();

AOCE_VE_EXPORT ITLayer<CrosshatchParamet>* createCrosshatchLayer();

AOCE_VE_EXPORT ITLayer<ColorMatrixParamet>* createColorMatrixLayer();

AOCE_VE_EXPORT ITLayer<FASTFeatureParamet>* createColourFASTFeatureDetector();

AOCE_VE_EXPORT ITLayer<CropParamet>* createCropLayer();

AOCE_VE_EXPORT ITLayer<BulrPositionParamet>* createBlurPositionLayer();

AOCE_VE_EXPORT ITLayer<BlurSelectiveParamet>* createBlurSelectiveLayer();

AOCE_VE_EXPORT BaseLayer* createDarkenBlendLayer();

AOCE_VE_EXPORT BaseLayer* createDifferenceBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createDissolveBlendLayer();

AOCE_VE_EXPORT BaseLayer* createDivideBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createEmbossLayer();

AOCE_VE_EXPORT BaseLayer* createExclusionBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createExposureLayer();

AOCE_VE_EXPORT ITLayer<FalseColorParamet>* createFalseColorLayer();

AOCE_VE_EXPORT ITLayer<float>* createGammaLayer();

AOCE_VE_EXPORT ITLayer<SphereRefractionParamet>* createSphereRefractionLayer();

AOCE_VE_EXPORT ITLayer<SphereRefractionParamet>* createGlassSphereLayer();

AOCE_VE_EXPORT ITLayer<PixellateParamet>* createHalftoneLayer();

AOCE_VE_EXPORT ITLayer<PixellateParamet>* createPixellateLayer();

AOCE_VE_EXPORT BaseLayer* createHardLightBlendLayer();

AOCE_VE_EXPORT ITLayer<HazeParamet>* createHazeLayer();

AOCE_VE_EXPORT ITLayer<float>* createLowPassLayer();

AOCE_VE_EXPORT ITLayer<float>* createHighPassLayer();

AOCE_VE_EXPORT HSBLayer* createHSBLayer();

AOCE_VE_EXPORT BaseLayer* createHueBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createHueLayer();

AOCE_VE_EXPORT ITLayer<HighlightShadowParamet>* createHighlightShadowLayer();

AOCE_VE_EXPORT ITLayer<HighlightShadowTintParamet>*
createHighlightShadowTintLayer();

AOCE_VE_EXPORT BaseLayer* createHistogramLayer(bool bSingle = true);

AOCE_VE_EXPORT ITLayer<IOSBlurParamet>* createIOSBlurLayer();

AOCE_VE_EXPORT ITLayer<uint32_t>* createKuwaharaLayer();

AOCE_VE_EXPORT BaseLayer* createLaplacianLayer(bool bsamll);

AOCE_VE_EXPORT ITLayer<LevelsParamet>* createLevelsLayer();

AOCE_VE_EXPORT BaseLayer* createLightenBlendLayer();

AOCE_VE_EXPORT BaseLayer* createLinearBurnBlendLayer();

AOCE_VE_EXPORT BaseLayer* createLuminosityBlendLayer();

AOCE_VE_EXPORT BaseLayer* createLuminanceLayer();

AOCE_VE_EXPORT BaseLayer* createMaskLayer();

AOCE_VE_EXPORT ITLayer<uint32_t>* createMedianLayer(bool bSingle = true);

AOCE_VE_EXPORT BaseLayer* createMedianK3Layer(bool bSingle = true);

AOCE_VE_EXPORT ITLayer<MonochromeParamet>* createMonochromeLayer();

AOCE_VE_EXPORT ITLayer<MotionBlurParamet>* createMotionBlurLayer();

AOCE_VE_EXPORT MotionDetectorLayer* createMotionDetectorLayer();

AOCE_VE_EXPORT BaseLayer* createMultiplyBlendLayer();

AOCE_VE_EXPORT ITLayer<HarrisCornerDetectionParamet>*
createNobleCornerDetectionLayer();

AOCE_VE_EXPORT BaseLayer* createNormalBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createOpacityLayer();

AOCE_VE_EXPORT BaseLayer* createOverlayBlendLayer();

AOCE_VE_EXPORT BaseLayer* createAlphaShowLayer();

AOCE_VE_EXPORT BaseLayer* createAlphaShow2Layer();

AOCE_VE_EXPORT BaseLayer* createConvertImageLayer();

}  // namespace vulkan
}  // namespace aoce

// struct SphereGlassParamet {
//     float aspectRatio = 1.0f;
//     float centerX = 0.5f;
//     float centerY = 0.5f;
//     float radius = 0.25f;
//     float refractiveIndex = 0.71f;
//     vec3 lightPosition = {-0.5, 0.5, 1.0};
//     vec3 ambientLightPosition = {0.0, 0.0, 1.0};
//     inline bool operator==(const SphereGlassParamet& right) {
//         return this->aspectRatio == right.aspectRatio &&
//                this->centerX == right.centerX &&
//                this->centerY == right.centerY && this->radius ==
//                right.radius
//                && this->refractiveIndex == refractiveIndex &&
//                this->lightPosition == right.lightPosition &&
//                this->ambientLightPosition == right.ambientLightPosition;
//     }
// };
