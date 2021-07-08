#include "VkExtraModule.hpp"

#include "../aoce/metadata/LayerMetadata.hpp"
#include "AoceVkExtra.h"

namespace aoce {
namespace vulkan {

void addGaussianMetadata(BGroupMetadata* gmd, const char* parametName,
                         const char* parametText,
                         const GaussianBlurParamet& paramet = {}) {
    auto ld =
        gmd->addGroupMetadata(parametName, "GaussianBlurParamet", parametText);
    ld->addMetadata("blurRadius", "BlurRadius", paramet.blurRadius, 1, 16);
}
void addSelectiveMetadata(BGroupMetadata* gmd, const char* parametName,
                          const char* parametText,
                          const SelectiveParamet& paramet = {}) {
    auto ld =
        gmd->addGroupMetadata(parametName, "SelectiveParamet", parametText);
    ld->addMetadata("aspectRatio", "AspectRatio", paramet.aspectRatio, 0.0f,
                    3.0f);
    ld->addUVMetadate("center", "Center", paramet.center);
    ld->addMetadata("radius", "Radius", paramet.radius, 0.0f, 1.0f);
    ld->addMetadata("size", "Size", paramet.size, 0.0f, 1.0f);
}
void addPositionMetadata(BGroupMetadata* gmd, const char* parametName,
                         const char* parametText,
                         const PositionParamet& paramet = {}) {
    auto ld =
        gmd->addGroupMetadata(parametName, "PositionParamet", parametText);
    ld->addMetadata("aspectRatio", "AspectRatio", paramet.aspectRatio, 0.0f,
                    3.0f);
    ld->addUVMetadate("center", "Center", paramet.center);
    ld->addMetadata("radius", "Radius", paramet.radius, 0.0f, 1.0f);
}
void addHarrisMetadata(BGroupMetadata* gmd, const char* parametName,
                       const char* parametText,
                       const HarrisDetectionBaseParamet& paramet = {}) {
    auto ld =
        gmd->addGroupMetadata(parametName, "GaussianBlurParamet", parametText);
    ld->addMetadata("edgeStrength", "EdgeStrength", paramet.edgeStrength, 0.1f,
                    5.0f);
    ld->addMetadata("threshold", "threshold", paramet.threshold, 0.0f, 1.0f);
    addGaussianMetadata(ld, "blueParamet", "Blur", paramet.blueParamet);
}
void addToonMetadata(BGroupMetadata* gmd, const char* parametName,
                     const char* parametText, const ToonParamet& paramet = {}) {
    auto ld = gmd->addGroupMetadata(parametName, "ToonParamet", parametText);
    ld->addMetadata("quantizationLevels", "Levels", paramet.quantizationLevels,
                    1.f, 100.0f);
    ld->addMetadata("threshold", "Threshold", paramet.threshold, 0.0f, 1.0f);
}

VkExtraModule::VkExtraModule() {}

VkExtraModule::~VkExtraModule() {}

bool VkExtraModule::loadModule() {
    auto& lm = LayerMetadataManager::Get();
    BGroupMetadata* layerMd = nullptr;
    lm.addMetadata("brightnessLayer", "Brightness", 0.0f, -1.0, 1.0f);
    lm.addMetadata("exposureLayer", "Exposure", 0.0f, -10.0, 10.0f);
    lm.addMetadata("contrastLayer", "Contrast", 1.0f, 0.0, 4.0f);
    lm.addMetadata("saturationLayer", "Saturation", 1.0f, 0.0, 2.0f);
    lm.addMetadata("gammaLayer", "Gamma", 1.0f, 0.0f, 3.0f);
    lm.addMetadata("solarizeLayer", "Solarize", 0.5f, 0.0f, 1.0f);
    // levelsLayer
    layerMd = lm.addGroupMetadata("levelsLayer", "LevelsParamet", "Levenls");
    layerMd->addColorMetadate("minVec", "Min", {0.0, 0.0, 0.0});
    layerMd->addColorMetadate("gammaVec", "Gamma", {1.0, 1.0, 1.0});
    layerMd->addColorMetadate("maxVec", "Max", {1.0, 1.0, 1.0});
    layerMd->addColorMetadate("minOut", "MinOut", {0.0, 0.0, 0.0});
    layerMd->addColorMetadate("maxOut", "MaxOut", {1.0, 1.0, 1.0});
    // rgbLayer
    layerMd = lm.addGroupMetadata("rgbLayer", "vec3", "RGB");
    layerMd->addMetadata("x", "R", 0.0f, 0.0f, 1.0f);
    layerMd->addMetadata("y", "G", 0.0f, 0.0f, 1.0f);
    layerMd->addMetadata("z", "B", 0.0f, 0.0f, 1.0f);
    // hueLayer
    lm.addMetadata("hueLayer", "Hue", 90.0f, 0.0, 360.0f);
    // vibranceLayer
    lm.addMetadata("vibranceLayer", "Vibrance", 0.0f, -1.2f, 1.2f);
    // whiteBalanceLayer
    layerMd = lm.addGroupMetadata("whiteBalanceLayer", "WhiteBalanceParamet",
                                  "WhiteBalance");
    layerMd->addMetadata("temperature", "Temperature", 5000.0f, 3000.0f,
                         8000.0f);
    layerMd->addMetadata("tint", "Tint", 0.0f, -200.0f, 200.0f);
    // highlightShadowLayer
    layerMd = lm.addGroupMetadata("highlightShadowLayer",
                                  "HighlightShadowParamet", "HighlightShadow");
    layerMd->addMetadata("shadows", "Shadows", 0.0f, 0.0f, 1.0f);
    layerMd->addMetadata("highlights", "Highlights", 1.0f, 0.0f, 1.0f);
    // highlightShadowTintLayer
    layerMd =
        lm.addGroupMetadata("highlightShadowTintLayer",
                            "HighlightShadowTintParamet", "HighlightShadow");
    layerMd->addMetadata("shadowTintIntensity", "ShadowTintIntensity", 0.0f,
                         0.0f, 1.0f);
    layerMd->addMetadata("highlightTintIntensity", "HighlightTintIntensity",
                         1.0f, 0.0f, 1.0f);
    layerMd->addColorMetadate("shadowTintColor", "ShadowTintColor",
                              {1.0, 0.0, 0.0});
    layerMd->addColorMetadate("highlightTintColor", "HighlightTintColor",
                              {0.0f, 0.0f, 1.0f});
    // skinToneLayer
    layerMd =
        lm.addGroupMetadata("skinToneLayer", "SkinToneParamet", "SkinTone");
    layerMd->addMetadata("skinToneAdjust", "ToneAdjust", 0.0f, -3.0f, 3.0f);
    layerMd->addMetadata("skinHue", "Hue", 0.05f, 0.0f, 1.0f);
    layerMd->addMetadata("skinHueThreshold", "HueThreshold", 40.0f, 0.0f,
                         360.0f);
    layerMd->addMetadata("maxHueShift", "MaxHueShift", 0.25f, 0.0f, 1.0f);
    layerMd->addMetadata("maxSaturationShift", "MaxSaturationShift", 0.4f, 0.0f,
                         1.0f);
    layerMd->addMetadata("upperSkinToneColor", "Green/Orange", false);
    // monochromeLayer
    layerMd = lm.addGroupMetadata("monochromeLayer", "MonochromeParamet",
                                  "Monochrome");
    layerMd->addMetadata("intensity", "Intensity", 1.0f, 0.0f, 1.0f);
    layerMd->addColorMetadate("color", "Color", {0.6f, 0.45f, 0.3f});
    // falseColorLayer
    layerMd = lm.addGroupMetadata("falseColorLayer", "FalseColorParamet",
                                  "FalseColor");
    layerMd->addColorMetadate("firstColor", "First", {0.0f, 0.0f, 0.5f});
    layerMd->addColorMetadate("secondColor", "Second", {0.6f, 0.45f, 0.3f});
    // hazeLayer
    layerMd = lm.addGroupMetadata("hazeLayer", "HazeParamet", "Haze");
    layerMd->addMetadata("distance", "Distance", 0.0f, -3.0f, 3.0f);
    layerMd->addMetadata("slope", "Slope", 0.0f, -3.0f, 3.0f);
    // sepiaLayer
    lm.addMetadata("sepiaLayer", "Sepia", 1.0f, 0.0f, 1.0f);
    // luminanceThresholdLayer
    lm.addMetadata("luminanceThresholdLayer", "LuminanceThreshold", 0.5f, 0.0f,
                   1.0f);
    // adaptiveThresholdLayer
    layerMd =
        lm.addGroupMetadata("adaptiveThresholdLayer",
                            "AdaptiveThresholdParamet", "AdaptiveThreshold");
    layerMd->addMetadata("boxSize", "BoxSize", 10, 1, 31);
    layerMd->addMetadata("offset", "Offset", 0.05f, 0.0f, 0.2f);
    // averageLuminanceThresholdLayer
    lm.addMetadata("luminanceThresholdLayer", "LuminanceThreshold", 1.0f, 0.0f,
                   2.0f);
    // chromaKeyLayer
    layerMd =
        lm.addGroupMetadata("chromaKeyLayer", "ChromaKeyParamet", "ChromaKey");
    layerMd->addColorMetadate("chromaColor", "Chroma", {0.0f, 1.0f, 0.0f});
    layerMd->addMetadata("lumaMask", "LumaMask", 1.0f, 0.0f, 10.0f);
    layerMd->addMetadata("alphaCutoffMin", "AlphaCutoffMin", 0.2f, 0.0f, 1.0f);
    layerMd->addMetadata("alphaScale", "AlphaScale", 10.0f, 0.0f, 20.0f);
    layerMd->addMetadata("alphaExponent", "AlphaExponent", 0.1f, 0.0f, 1.0f);
    layerMd->addColorMetadate("ambientColor", "Ambient", {1.0f, 1.0f, 1.0f});
    layerMd->addMetadata("ambientScale", "AmbientScale", 0.0f, 0.0f, 1.0f);
    layerMd->addMetadata("despillScale", "DespillScale", 0.0f, 0.0f, 1.0f);
    layerMd->addMetadata("despillExponent", "DespillExponent", 0.1f, 0.0f,
                         1.0f);
    // sharpenLayer
    layerMd = lm.addGroupMetadata("sharpenLayer", "SharpenParamet", "Sharpen");
    layerMd->addMetadata("offset", "Offset", 1, 1, 5);
    layerMd->addMetadata("sharpness", "Sharpness", 0.0f, -4.0f, 4.0f);
    // unsharpMaskLayer
    layerMd = lm.addGroupMetadata("unsharpMaskLayer", "UnsharpMaskParamet",
                                  "UnsharpMask");
    layerMd->addMetadata("intensity", "Intensity", 1.0f, 0.0f, 1.0f);
    addGaussianMetadata(layerMd, "blur", "Blur", {4, 0.0f});
    // gaussianBlurLayer
    layerMd = lm.addGroupMetadata("gaussianBlurLayer", "GaussianBlurParamet",
                                  "GaussianBlur");
    layerMd->addMetadata("blurRadius", "BlurRadius", 3, 1, 16);
    // boxBlurLayer
    layerMd =
        lm.addGroupMetadata("boxBlurLayer", "KernelSizeParamet", "KernelSize");
    layerMd->addMetadata("kernelSizeX", "SizeX", 3, 1, 32);
    layerMd->addMetadata("kernelSizeY", "SizeY", 3, 1, 32);
    // blurSelectiveLayer
    layerMd = lm.addGroupMetadata("blurSelectiveLayer", "BlurSelectiveParamet",
                                  "BlurSelective");
    addGaussianMetadata(layerMd, "gaussian", "Gaussian");
    addSelectiveMetadata(layerMd, "blurPosition", "BlurPosition");
    // blurPositionLayer
    layerMd = lm.addGroupMetadata("blurPositionLayer", "BlurPositionParamet",
                                  "BlurPosition");
    addGaussianMetadata(layerMd, "gaussian", "Gaussian");
    addPositionMetadata(layerMd, "blurPosition", "BlurPosition");
    // iosBlurLayer
    layerMd = lm.addGroupMetadata("iosBlurLayer", "IOSBlurParamet", "IOSBlur");
    layerMd->addMetadata("sacle", "Scale", 4.0f, 1.0f, 10.f);
    layerMd->addMetadata("saturation", "Saturation", 0.8f, 0.0f, 1.f);
    layerMd->addMetadata("range", "Range", 0.6f, 0.0f, 1.f);
    addGaussianMetadata(layerMd, "blurParamet", "Blur");
    // bilateralLayer
    layerMd =
        lm.addGroupMetadata("bilateralLayer", "BilateralParamet", "Bilateral");
    layerMd->addMetadata("kernelSize", "Size", 3, 1, 20);
    layerMd->addMetadata("sigma_spatial", "Spatial", 10.f, 1.0f, 100.f);
    layerMd->addMetadata("sigma_spatial", "Color", 10.f, 1.0f, 100.f);
    // tiltShiftLayer
    layerMd =
        lm.addGroupMetadata("tiltShiftLayer", "TiltShiftParamet", "TiltShift");
    layerMd->addMetadata("topFocusLevel", "Top", 0.4f, 0.0f, 1.0f);
    layerMd->addMetadata("bottomFocusLevel", "Bottom", 0.6f, 0.0f, 1.0f);
    layerMd->addMetadata("focusFallOffRate", "Rate", 0.2f, 0.0f, 1.0f);
    addGaussianMetadata(layerMd, "blur", "Blur");
    // sobelEdgeDetectionLayer/prewittEdgeDetectionLayer
    lm.addMetadata("sobelEdgeDetectionLayer", "EdgeScale", 1.0f, 0.1f, 10.f);
    lm.addMetadata("prewittEdgeDetectionLayer", "EdgeScale", 1.0f, 0.1f, 10.f);
    // thresholdEdgeDetectionLayer
    layerMd = lm.addGroupMetadata("thresholdEdgeDetectionLayer",
                                  "ThresholdSobelParamet", "ThresholdSobel");
    layerMd->addMetadata("edgeStrength", "EdgeStrength", 1.0f, 0.1f, 10.0f);
    layerMd->addMetadata("threshold", "Threshold", 0.25f, 0.0f, 1.0f);
    // cannyEdgeDetectionLayer
    layerMd =
        lm.addGroupMetadata("cannyEdgeDetectionLayer",
                            "CannyEdgeDetectionParamet", "CannyEdgeDetection");
    layerMd->addMetadata("minThreshold", "min", 0.1f, 0.0f, 1.0f);
    layerMd->addMetadata("maxThreshold", "max", 0.45f, 0.0f, 1.0f);
    addGaussianMetadata(layerMd, "blueParamet", "Blur");
    // harrisCornerDetectionLayer
    layerMd = lm.addGroupMetadata("harrisCornerDetectionLayer",
                                  "HarrisCornerDetectionParamet",
                                  "HarrisCornerDetection");
    layerMd->addMetadata("harris", "Harris", 0.04f, 0.01f, 0.5f);
    layerMd->addMetadata("sensitivity", "Sensitivity", 5.0f, 1.0f, 10.0f);
    addHarrisMetadata(layerMd, "harrisBase", "Harris");
    // nobleCornerDetectionLayer
    layerMd = lm.addGroupMetadata("nobleCornerDetectionLayer",
                                  "NobleCornerDetectionParamet",
                                  "NobleCornerDetection");
    layerMd->addMetadata("sensitivity", "Sensitivity", 5.0f, 1.0f, 10.0f);
    addHarrisMetadata(layerMd, "harrisBase", "Harris");
    // shiTomasiDetectionLayer
    layerMd = lm.addGroupMetadata("shiTomasiDetectionLayer",
                                  "NobleCornerDetectionParamet",
                                  "NobleCornerDetection");
    layerMd->addMetadata("sensitivity", "Sensitivity", 5.0f, 1.0f, 10.0f);
    addHarrisMetadata(layerMd, "harrisBase", "Harris");
    // fastFeatureLayer
    layerMd = lm.addGroupMetadata("fastFeatureLayer", "FASTFeatureParamet",
                                  "FASTFeature");
    layerMd->addMetadata("boxSize", "Size", 3, 1, 20);
    layerMd->addMetadata("offset", "Offset", 1.f, 0.1f, 10.f);
    // dilation/erosion/closing/opening
    lm.addMetadata("dilationLayer", "Size", 3, 1, 32);
    lm.addMetadata("erosionLayer", "Size", 3, 1, 32);
    lm.addMetadata("closingLayer", "Size", 3, 1, 32);
    lm.addMetadata("openingLayer", "Size", 3, 1, 32);
    // low/high
    lm.addMetadata("lowPassLayer", "Mix", 0.5f, 0.0f, 1.0f);
    lm.addMetadata("highPassLayer", "Mix", 0.5f, 0.0f, 1.0f);
    lm.addMetadata("motionDetectorLayer", "Mix", 0.5f, 0.0f, 1.0f);
    // motionBlurLayer
    layerMd = lm.addGroupMetadata("motionBlurLayer", "MotionBlurParamet",
                                  "MotionBlur");
    layerMd->addMetadata("blurSize", "Size", 1.0f, 0.0f, 5.0f);
    layerMd->addMetadata("blurAngle", "Angle", 0.f, 0.0f, 360.f);
    // zoomBlurLayer
    layerMd =
        lm.addGroupMetadata("zoomBlurLayer", "ZoomBlurParamet", "ZoomBlur");
    layerMd->addUVMetadate("blurCenter", "Center", {0.5f, 0.5f});
    layerMd->addMetadata("blurSize", "Size", 1.0f, 0.0f, 5.0f);
    // dissolveBlendLayer/alphaBlendLayer
    lm.addMetadata("dissolveBlendLayer", "Mix", 0.5f, 0.0f, 1.0f);
    lm.addMetadata("alphaBlendLayer", "Mix", 0.5f, 0.0f, 1.0f);
    // poissonLayer
    layerMd = lm.addGroupMetadata("poissonLayer", "PoissonParamet", "Poisson");
    layerMd->addMetadata("iterationNum", "Num", 10, 1, 1000);
    layerMd->addMetadata("percent", "Mix", 0.5f, 0.0f, 1.0f);
    // pixellateLayer
    layerMd =
        lm.addGroupMetadata("pixellateLayer", "PixellateParamet", "Pixellate");
    layerMd->addMetadata("fractionalWidthOfPixel", "Pixel", 0.05f, 0.0f, 1.0f);
    layerMd->addMetadata("aspectRatio", "AspectRatio", 0.5625f, 0.0f, 3.0f);
    // polarPixellateLayer
    layerMd = lm.addGroupMetadata("polarPixellateLayer",
                                  "PolarPixellateParamet", "PolarPixellate");
    layerMd->addUVMetadate("center", "Center", {0.5f, 0.5f});
    layerMd->addUVMetadate("size", "PixelSize", {0.05f, 0.05f});
    // pixellatePositionLayer
    layerMd = lm.addGroupMetadata("pixellatePositionLayer", "SelectiveParamet",
                                  "Selective");
    layerMd->addMetadata("aspectRatio", "AspectRatio", 0.5625f, 0.0f, 3.0f);
    layerMd->addUVMetadate("center", "Center", {0.5f, 0.5f});
    layerMd->addMetadata("radius", "Radius", 0.25f, 0.0f, 1.0f);
    layerMd->addMetadata("size", "Size", 0.125f, 0.0f, 1.0f);
    // polkaDotLayer
    layerMd =
        lm.addGroupMetadata("polkaDotLayer", "PolkaDotParamet", "PolkaDot");
    layerMd->addMetadata("dotScaling", "Scale", 0.9, 0.0f, 1.0f);
    layerMd->addMetadata("fractionalWidthOfPixel", "PixelSize", 0.05, 0.0f,
                         1.0f);
    layerMd->addMetadata("aspectRatio", "AspectRatio", 0.5625f, 0.0f, 3.0f);
    // crosshatchLayer
    layerMd = lm.addGroupMetadata("crosshatchLayer", "CrosshatchParamet",
                                  "Crosshatch");
    layerMd->addMetadata("crossHatchSpacing", "Spacing", 0.03, 0.0f, 1.0f);
    layerMd->addMetadata("lineWidth", "LineWidth", 0.003, 0.0f, 0.1f);
    // sketchLayer
    lm.addMetadata("sketchLayer", "EdgeStrength", 1.0f, 0.1f, 10.0f);
    // thresholdSketchLayer
    layerMd = lm.addGroupMetadata("thresholdSketchLayer",
                                  "ThresholdSobelParamet", "ThresholdSobel");
    layerMd->addMetadata("edgeStrength", "EdgeStrength", 1.0f, 0.1f, 10.0f);
    layerMd->addMetadata("threshold", "Threshold", 0.25f, 0.0f, 1.0f);
    // toonLayer
    layerMd = lm.addGroupMetadata("toonLayer", "ToonParamet", "Toon");
    layerMd->addMetadata("quantizationLevels", "Levels", 10.0f, 1.f, 256.0f);
    layerMd->addMetadata("threshold", "Threshold", 0.2f, 0.0f, 1.0f);
    // smoothToonLayer
    layerMd = lm.addGroupMetadata("smoothToonLayer", "SmoothToonParamet",
                                  "SmoothToon");
    addGaussianMetadata(layerMd, "blur", "Blur");
    addToonMetadata(layerMd, "toon", "Toon");
    // embossLayer
    lm.addMetadata("embossLayer", "Strength", 0.4f, 0.1f, 1.0f);
    // posterizeLayer
    lm.addMetadata("posterizeLayer", "Levels", 10, 1, 256);
    // swirlLayer
    layerMd = lm.addGroupMetadata("swirlLayer", "SwirlParamet", "Swirl");
    layerMd->addUVMetadate("center", "Center", {0.5f, 0.5f});
    layerMd->addMetadata("radius", "Radius", 0.5f, 0.0f, 1.0f);
    layerMd->addMetadata("angle", "Angle", 1.0f, 0.0f, 5.0f);
    // bulgeDistortionLayer
    layerMd = lm.addGroupMetadata("bulgeDistortionLayer", "DistortionParamet",
                                  "Distortion");
    layerMd->addMetadata("aspectRatio", "AspectRatio", 0.5625f, 0.0f, 3.0f);
    layerMd->addUVMetadate("center", "Center", {0.5f, 0.5f});
    layerMd->addMetadata("radius", "Radius", 0.25f, 0.0f, 1.0f);
    layerMd->addMetadata("scale", "Scale", 0.5f, -1.0f, 1.0f);
    // stretchDistortionLayer
    layerMd =
        lm.addGroupMetadata("stretchDistortionLayer", "vec2", "Distortion");
    layerMd->addMetadata("x", "U", 0.5f, 0.0f, 1.0f);
    layerMd->addMetadata("y", "V", 0.5f, 0.0f, 1.0f);
    // sphereRefractionLayer
    layerMd = lm.addGroupMetadata(
        "sphereRefractionLayer", "SphereRefractionParamet", "SphereRefraction");
    layerMd->addMetadata("aspectRatio", "AspectRatio", 0.5625f, 0.0f, 3.0f);
    layerMd->addUVMetadate("center", "Center", {0.5f, 0.5f});
    layerMd->addMetadata("radius", "Radius", 0.25f, 0.0f, 1.0f);
    layerMd->addMetadata("refractiveIndex", "RefractiveIndex", 0.71f, 0.0f,
                         1.0f);
    // vignetteLayer
    layerMd =
        lm.addGroupMetadata("vignetteLayer", "VignetteParamet", "Vignette");
    layerMd->addUVMetadate("vignetteCenter", "Center", {0.5f, 0.5f});
    layerMd->addColorMetadate("vignetteColor", "Color", {0.0f, 0.0f, 0.0f});
    layerMd->addMetadata("vignetteStart", "Start", 0.3f, 0.0f, 1.0f);
    layerMd->addMetadata("vignetteEnd", "End", 0.75f, 0.0f, 1.0f);
    return true;
}

void VkExtraModule::unloadModule() {}

ADD_MODULE(VkExtraModule, aoce_vulkan_extra)

}  // namespace vulkan
}  // namespace aoce