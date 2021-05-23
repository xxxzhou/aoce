#pragma once

#include "../aoce/AoceCore.h"

#ifdef _WIN32
#if defined(AOCE_VULKAN_EXTRA_EXPORT_DEFINE)
#define AOCE_VE_EXPORT __declspec(dllexport)
#else
#define AOCE_VE_EXPORT __declspec(dllimport)
#endif
#elif __ANDROID__
#if defined(AOCE_VULKAN_EXTRA_EXPORT_DEFINE)
#define AOCE_VE_EXPORT __attribute__((visibility("default")))
#else
#define AOCE_VE_EXPORT
#endif
#endif

namespace aoce {

enum class ConvertType : int32_t { other = 0, rgba82rgba32f, rgba32f2rgba8 };

enum class ReduceOperate : int32_t {
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
    // sigma值越小,整个分布长度范围越大,原始值占比越高,周围占比越低
    // 如果为0,根据blurRadius自动计算
    float sigma = 0.0f;

    inline bool operator==(const GaussianBlurParamet& right) {
        return this->blurRadius == right.blurRadius &&
               this->sigma == right.sigma;
    }
};

// https://www.unrealengine.com/en-US/tech-blog/setting-up-a-chroma-key-material-in-ue4
struct ChromaKeyParamet {
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
    inline bool operator==(const ChromaKeyParamet& right) {
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

// 确定像素周围的局部亮度,比较周边与局部亮度
struct AdaptiveThresholdParamet {
    // 背景平均模糊半径
    int32_t boxSize = 10;
    // 比较平均亮度偏移
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
    // 检测到一个点作为拐角的阈值.
    // 根据尺寸,光线条件和iOS设备相机类型的不同,此方法可能会有很大的不同
    // 因此可能需要一些试验才能确定适合您的情况.默认值为0.20.
    float threshold = 0.2f;

    inline bool operator==(const HarrisDetectionBaseParamet& right) {
        return this->edgeStrength == right.edgeStrength &&
               this->blueParamet == right.blueParamet &&
               this->threshold == right.threshold;
    }
};

// Harris角点检测
struct HarrisCornerDetectionParamet {
    HarrisDetectionBaseParamet harrisBase = {};
    float harris = 0.04f;
    // 一个内部比例因子,用于调整在滤镜中生成的边角图的动态范围.默认值为5.0.
    float sensitivity = 5.0f;
};

struct NobleCornerDetectionParamet {
    HarrisDetectionBaseParamet harrisBase = {};
    float sensitivity = 5.0f;
};

// 执行Canny边缘阈值检测
struct CannyEdgeDetectionParamet {
    GaussianBlurParamet blueParamet = {4, 0.0f};
    // 任何梯度幅度大于此阈值的边都将通过并显示在最终结果中
    float minThreshold = 0.1f;
    // 任何梯度幅度低于此阈值的边将失败,并从最终结果中删除.
    float maxThreshold = 0.4f;
};

struct FASTFeatureParamet {
    int32_t boxSize = 5;
    float offset = 1.0f;
};

// 双边滤波
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

// 在图像上创建凸出的失真
struct DistortionParamet {
    float aspectRatio = 0.5625f;
    // 图像的中心(在0-1.0的标准化坐标中),默认值为(0.5,0.5)
    vec2 center = {0.5f, 0.5f};
    // 从中心开始应用变形的半径,默认值为0.25
    float radius = 0.25f;
    // 要应用的失真量,从-1.0到1.0,默认值为0.5
    float scale = 0.5f;

    inline bool operator==(const DistortionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius &&
               this->scale == right.scale;
    }
};

// 圆形区域
struct PositionParamet {
    // 图像的纵横比,如果想要圆形,height/width
    float aspectRatio = 0.5625f;
    // 圆形区域的中心
    vec2 center = {0.5f, 0.5f};
    // 圆形区域的半径
    float radius = 0.25f;

    inline bool operator==(const PositionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius;
    }
};

// 圆形区域
struct SelectiveParamet {
    // 图像的纵横比,如果想要圆形,height/width
    float aspectRatio = 0.5625f;
    // 圆形区域的中心
    vec2 center = {0.5f, 0.5f};
    // 圆形区域的半径
    float radius = 0.25f;
    // 圆形区域的大小
    float size = 0.125f;
    inline bool operator==(const SelectiveParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius &&
               this->size == size;
    }
};

// 圆形区域模糊
struct BulrPositionParamet {
    GaussianBlurParamet gaussian = {};
    PositionParamet bulrPosition = {};
};

// 圆形区域不模糊
struct BlurSelectiveParamet {
    GaussianBlurParamet gaussian = {};
    SelectiveParamet bulrPosition = {};
};

struct SphereRefractionParamet {
    // 图像的纵横比,如果想要圆形,height/width
    float aspectRatio = 0.5625f;
    vec2 center = {0.5f, 0.5f};
    float radius = 0.25f;
    float refractiveIndex = 0.71f;
    inline bool operator==(const SphereRefractionParamet& right) {
        return this->aspectRatio == right.aspectRatio &&
               this->center == right.center && this->radius == right.radius &&
               this->refractiveIndex == refractiveIndex;
    }
};

// 对图像或视频应用像素化/半色调效果,如马赛克/新闻打印
struct PixellateParamet {
    // 像素的大小,以图像的宽度和高度的分数为单位(0.0-1.0,默认为0.05)
    float fractionalWidthOfPixel = 0.01f;
    // 图像的纵横比,如果想要圆形,height/width
    float aspectRatio = 0.5625f;
    inline bool operator==(const PixellateParamet& right) {
        return this->fractionalWidthOfPixel == right.fractionalWidthOfPixel &&
               this->aspectRatio == right.aspectRatio;
    }
};

// 通过将矩阵应用于图像来变换图像的颜色
struct ColorMatrixParamet {
    // 新的变换后的颜色替换每个像素的原始颜色的程度
    float intensity = 1.0f;
    // 用于转换图像中每种颜色的4x4矩阵
    Mat4x4 mat = {};

    inline bool operator==(const ColorMatrixParamet& right) {
        return this->intensity == right.intensity && this->mat == right.mat;
    }
};

// 裁剪图像的特定区域
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

// 根据图像的亮度在两种用户指定的颜色之间进行混合
struct FalseColorParamet {
    // 暗区
    vec3 firstColor = {0.0f, 0.0f, 0.5f};
    // 亮区
    vec3 secondColor = {1.0f, 0.0f, 0.0f};
    inline bool operator==(const FalseColorParamet& right) {
        return this->firstColor == right.firstColor &&
               this->secondColor == right.secondColor;
    }
};

// 用于添加或删除雾度(类似于UV滤镜)
struct HazeParamet {
    // 所应用颜色的强度.默认值为0.最好是-.3和.3之间的值.
    float distance = 0.0f;
    // 颜色变化的量.默认值为0.最好是-.3和.3之间的值.
    float slope = 0.0f;
    inline bool operator==(const HazeParamet& right) {
        return this->distance == right.distance && this->slope == right.slope;
    }
};

// 调整图像的阴影和高光
struct HighlightShadowParamet {
    // 增加阴影以使阴影变淡,从0.0到1.0,默认值为0.0.
    float shadows = 0.0f;
    // 从1.0降低到0.0,以1.0为默认值将高光变暗.
    float highlights = 1.0f;
    inline bool operator==(const HighlightShadowParamet& right) {
        return this->shadows == right.shadows &&
               this->highlights == right.highlights;
    }
};

// 使用颜色和强度独立地着色图像的阴影和高光
struct HighlightShadowTintParamet {
    // 阴影着色强度,从0.0到1.0.默认值:0.0.
    float shadowTintIntensity = 0.0f;
    // 阴影色调RGB颜色(GPUVector4).默认值:({1.0f, 0.0f, 0.0f, 1.0f}红色).
    vec3 shadowTintColor = {1.0f, 0.0f, 0.0f};
    // 突出显示色调强度,从0.0到1.0,默认值为0.0.
    float highlightTintIntensity = 1.0f;
    // highlightTintColor:高亮色调RGB颜色(GPUVector4).默认值:({0.0f,
    // 0.0f, 1.0f, 1.0f}蓝色).
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

// 类似Photoshop的色阶调整,所有参数在[0,1]范围内浮动
struct LevelsParamet {
    vec3 minVec = {0.0, 0.0, 0.0};
    vec3 gammaVec = {1.0, 1.0, 1.0};
    vec3 maxVec = {1.0, 1.0, 1.0};
    vec3 minOut = {0.0, 0.0, 0.0};
    vec3 maxOunt = {1.0, 1.0, 1.0};
};

// 根据每个像素的亮度将图像转换为单色版本
struct MonochromeParamet {
    // 特定颜色替换正常图像颜色的程度(0.0-1.0,默认值为1.0)
    float intensity = 1.0f;
    // 用作效果基础的颜色,默认为(0.6,0.45,0.3,1.0)
    vec3 color = {0.6f, 0.45f, 0.3f};
    inline bool operator==(const MonochromeParamet& right) {
        return this->intensity == right.intensity && this->color == right.color;
    }
};

// 对图像应用定向运动模糊
struct MotionBlurParamet {
    // 模糊大小的倍数,范围从0.0开始,默认为1.0
    float blurSize = 1.0f;
    // 模糊的角度方向,以度为单位.默认情况下为0度.
    float blurAngle = 0.0f;
    inline bool operator==(const MotionBlurParamet& right) {
        return this->blurSize == right.blurSize &&
               this->blurAngle == right.blurAngle;
    }
};

// 应用两个图像的泊松混合
struct PoissonParamet {
    // 混合范围从0.0(仅图像1)到1.0(仅图像2渐变),以1.0为正常水平
    float percent = 0.5f;
    // 播渐变的次数.如果您想获得接近收敛的效果,则可以将其提高到100甚至1000.是的,这会很慢.
    int32_t iterationNum = 10;
};

// 生成充满Perlin噪点的图像
struct PerlinNoiseParamet {
    // 产生的噪声的标度
    float scale = 8.0f;
    // 噪声颜色最小值
    vec4 colorStart = {0.0, 0.0, 0.0, 1.0};
    // 噪声颜色最大值
    vec4 colorFinish = {1.0, 1.0, 1.0, 1.0};
    inline bool operator==(const PerlinNoiseParamet& right) {
        return this->colorStart == right.colorStart &&
               this->colorFinish == right.colorFinish &&
               this->scale == right.scale;
    }
};

// 基于极坐标而不是笛卡尔坐标,对图像或视频应用像素化效果
struct PolarPixellateParamet {
    // 要应用像素化的中心
    vec2 center = {0.5f, 0.5f};
    // 像素大小,分为宽度和高度分量.默认值为(0.05,0.05)
    vec2 size = {0.05f, 0.05f};

    inline bool operator==(const PolarPixellateParamet& right) {
        return this->center == right.center && this->size == right.size;
    }
};

// 将图像分成规则网格内的彩色点
struct PolkaDotParamet {
    // 点在每个网格空间中所占的比例从0.0到1.0,默认值为0.9.
    float dotScaling = 0.90f;
    // 点的大小,以图像的宽度和高度的分数为单位(0.0-1.0,默认为0.05)
    float fractionalWidthOfPixel = 0.01f;
    float aspectRatio = 0.5625f;
    inline bool operator==(const PolkaDotParamet& right) {
        return this->dotScaling == right.dotScaling &&
               this->fractionalWidthOfPixel == right.fractionalWidthOfPixel &&
               this->aspectRatio == right.aspectRatio;
    }
};

// 图像锐化
struct SharpenParamet {
    int offset = 1;
    // 要应用的清晰度调整(-4.0-4.0,默认值为0.0)
    float sharpness = 0.0f;
    inline bool operator==(const SharpenParamet& right) {
        return this->offset == right.offset &&
               this->sharpness == right.sharpness;
    }
};

// 肤色调整滤镜,可影响浅肤色颜色的唯一范围,并相应地调整粉红色/绿色或粉红色/橙色范围
struct SkinToneParamet {
    // 调整肤色的量.默认值:0.0,建议的最小值/最大值:分别为-0.3和0.3.
    float skinToneAdjust = 0.0f;
    // 要检测的皮肤色调.默认值:0.05(白皙至泛红皮肤).
    float skinHue = 0.05f;
    // 皮肤色调的变化量.默认值:40.0.
    float skinHueThreshold = 40.0f;
    // 允许的最大色相偏移量.默认值:0.25
    float maxHueShift = 0.25f;
    // 要移动的最大饱和量(使用橙色时).默认值:0.4
    float maxSaturationShift = 0.4f;
    // Green/Orange[0,1]
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

// 使用Sobel边缘检测在对象周围放置黑色边框,然后对图像中存在的颜色进行量化,以使图像具有卡通般的质量
struct ToonParamet {
    // 边缘检测的灵敏度,值越小灵敏度越高.范围从0.0到1.0,默认值为0.2
    float threshold = 0.2f;
    // 要在最终图像中表示的色阶数.默认值为10.0
    float quantizationLevels = 10.0f;
    inline bool operator==(const ToonParamet& right) {
        return this->threshold == right.threshold &&
               this->quantizationLevels == right.quantizationLevels;
    }
};

// 这使用与ToonLayer相似的过程,只是它在卡通效果之前带有高斯模糊用来平滑噪声
struct SmoothToonParamet {
    GaussianBlurParamet blur = {};
    ToonParamet toon = {};
};

struct SoftEleganceParamet {
    GaussianBlurParamet blur = {10, 0.0f};
    float mix = 0.14f;
};

// 在图像上创建漩涡形失真
struct SwirlParamet {
    // 要绕图像扭曲的图像中心(以0-1.0的标准化坐标表示),默认值为(0.5,0.5)
    vec2 center = {0.5f, 0.5f};
    // 从中心开始应用变形的半径,默认值为0.5
    float radius = 0.5f;
    // 应用于图像的扭曲量,默认值为1.0
    float angle = 1.0f;
    inline bool operator==(const SwirlParamet& right) {
        return this->center == right.center && this->radius == right.radius &&
               this->angle == right.angle;
    }
};

// 执行Sobel边缘阈值检测
struct ThresholdSobelParamet {
    float edgeStrength = 1.0f;
    // 高于此阈值的任何边缘将为黑色,低于白色的任何边缘.
    float threshold = 0.25f;
    inline bool operator==(const ThresholdSobelParamet& right) {
        return this->edgeStrength == right.edgeStrength &&
               this->threshold == right.threshold;
    }
};

// 模拟倾斜移位镜头效果
struct TiltShiftParamet {
    GaussianBlurParamet blur = {7, 0.0f};
    // 图像中对焦区域顶部的标准化位置,范围0.0-1.0
    float topFocusLevel = 0.4f;
    // 图像中对焦区域底部的标准化位置,范围0.0-1.0,需要高于topFocusLevel
    float bottomFocusLevel = 0.6f;
    // 图像从对焦区域模糊的速率,默认为0.2
    float focusFallOffRate = 0.2f;
};

// 应用不清晰的蒙版
struct UnsharpMaskParamet {
    // 高斯模糊的模糊参数
    GaussianBlurParamet blur = {4, 0.0f};
    // 清晰度,(0.0-1.0),默认1.0
    float intensity = 1.0f;
};

// 执行渐晕效果,使边缘的图像淡化
struct VignetteParamet {
    // 小插图的中心,以tex坐标(CGPoint)为单位,默认值为(0.5,0.5)
    vec2 vignetteCenter = {0.5f, 0.5f};
    // 用于小插图(GPUVector3)的颜色,默认为黑色
    vec3 vignetteColor = {0.0f, 0.0f, 0.0f};
    // 距小插图效果开始的中心的标准化距离,默认值为0.5
    float vignetteStart = 0.3f;
    // 距小插图效果结束的中心的标准化距离,默认值为0.75
    float vignetteEnd = 0.75f;

    inline bool operator==(const VignetteParamet& right) {
        return this->vignetteCenter == right.vignetteCenter &&
               this->vignetteColor == right.vignetteColor &&
               this->vignetteStart == right.vignetteStart &&
               this->vignetteEnd == right.vignetteEnd;
    }
};

// 调整图像的白平衡.
struct WhiteBalanceParamet {
    // 调整图像所用的温度,以ºK为单位.值4000非常凉爽,而7000非常温暖.默认值为5000.
    float temperature = 5000.0f;
    // 用于调整图像的色调.值-200表示非常绿色,而200表示非常粉红色.默认值为0.
    float tint = 0.0f;
    inline bool operator==(const WhiteBalanceParamet& right) {
        return this->temperature == right.temperature &&
               this->tint == right.tint;
    }
};

// 将定向运动模糊应用于图像
struct ZoomBlurParamet {
    // 模糊中心.默认为(0.5,0.5)
    vec2 blurCenter = {0.5f, 0.5f};
    // 模糊大小的倍数,范围从0.0开始,默认为1.0
    float blurSize = 1.0f;
    inline bool operator==(const ZoomBlurParamet& right) {
        return this->blurCenter == right.blurCenter &&
               this->blurSize == right.blurSize;
    }
};

typedef ITLayer<SoftEleganceParamet> ASoftEleganceLayer;
typedef ITLayer<PerlinNoiseParamet> APerlinNoiseLayer;
typedef ITLayer<float> AFloatLayer;

typedef ITLayer<vec2> IStretchDistortionLayer;
typedef ITLayer<vec3> IRGBLayer;
typedef ITLayer<uint32_t> IMedianLayer;
typedef ITLayer<Mat3x3> I3x3ConvolutionLayer;
typedef ITLayer<int32_t> IMorphLayer;

typedef ITLayer<float> IBrightnessLayer;
typedef ITLayer<float> IExposureLayer;
typedef ITLayer<float> IContrastLayer;
typedef ITLayer<float> ISaturationLayer;
typedef ITLayer<float> IGammaLayer;
typedef ITLayer<float> ISolarizeLayer;
typedef ITLayer<float> IHueLayer;
typedef ITLayer<float> IVibranceLayer;
typedef ITLayer<float> ISepiaLayer;
typedef ITLayer<float> IOpacityLayer;
typedef ITLayer<float> ILuminanceThresholdLayer;
typedef ITLayer<float> IAverageLuminanceThresholdLayer;

typedef ITLayer<SizeScaleParamet> ISizeScaleLayer;
typedef ITLayer<KernelSizeParamet> IKernelSizeLayer;
typedef ITLayer<GaussianBlurParamet> IGaussianBlurLayer;
typedef ITLayer<ChromaKeyParamet> IChromaKeyLayer;
typedef ITLayer<AdaptiveThresholdParamet> IAdaptiveThresholdLayer;
typedef ITLayer<GuidedParamet> IGuidedLayer;
typedef ITLayer<HarrisCornerDetectionParamet> IHarrisCornerDetectionLayer;
typedef ITLayer<NobleCornerDetectionParamet> INobleCornerDetectionLayer;
typedef ITLayer<CannyEdgeDetectionParamet> ICannyEdgeDetectionLayer;
typedef ITLayer<FASTFeatureParamet> IFASTFeatureLayer;
typedef ITLayer<BilateralParamet> IBilateralLayer;
typedef ITLayer<DistortionParamet> IDistortionLayer;
typedef ITLayer<PositionParamet> IPositionLayer;
typedef ITLayer<SelectiveParamet> ISelectiveLayer;
typedef ITLayer<BulrPositionParamet> IBulrPositionLayer;
typedef ITLayer<BlurSelectiveParamet> IBlurSelectiveLayer;
typedef ITLayer<SphereRefractionParamet> ISphereRefractionLayer;
typedef ITLayer<PixellateParamet> IPixellateLayer;
typedef ITLayer<ColorMatrixParamet> IColorMatrixLayer;
typedef ITLayer<CropParamet> ICropLayer;
typedef ITLayer<CrosshatchParamet> ICrosshatchLayer;
typedef ITLayer<FalseColorParamet> IFalseColorLayer;
typedef ITLayer<HazeParamet> IHazeLayer;
typedef ITLayer<HighlightShadowParamet> IHighlightShadowLayer;
typedef ITLayer<HighlightShadowTintParamet> IHighlightShadowTintLayer;
typedef ITLayer<IOSBlurParamet> IIOSBlurLayer;
typedef ITLayer<LevelsParamet> ILevelsLayer;
typedef ITLayer<MonochromeParamet> IMonochromeLayer;
typedef ITLayer<MotionBlurParamet> IMotionBlurLayer;
typedef ITLayer<PoissonParamet> IPoissonLayer;
typedef ITLayer<PolarPixellateParamet> IPolarPixellateLayer;
typedef ITLayer<PolkaDotParamet> IPolkaDotLayer;
typedef ITLayer<SharpenParamet> ISharpenLayer;
typedef ITLayer<SkinToneParamet> ISkinToneLayer;
typedef ITLayer<ToonParamet> IToonLayer;
typedef ITLayer<SmoothToonParamet> ISmoothToonLayer;
typedef ITLayer<SwirlParamet> ISwirlParametLayer;
typedef ITLayer<SharpenParamet> ISharpenParametLayer;
typedef ITLayer<ThresholdSobelParamet> IThresholdSobelLayer;
typedef ITLayer<TiltShiftParamet> ITiltShiftLayer;
typedef ITLayer<UnsharpMaskParamet> IUnsharpMaskLayer;
typedef ITLayer<VignetteParamet> IVignetteLayer;
typedef ITLayer<WhiteBalanceParamet> IWhiteBalanceLayer;
typedef ITLayer<ZoomBlurParamet> IZoomBlurLayer;

}  // namespace aoce