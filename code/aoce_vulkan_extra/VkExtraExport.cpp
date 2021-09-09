#include "VkExtraExport.h"

#include "layer/VkAdaptiveThresholdLayer.hpp"
#include "layer/VkAlphaShowLayer.hpp"
#include "layer/VkBilateralLayer.hpp"
#include "layer/VkBlendingModeLayer.hpp"
#include "layer/VkBlurBlendBaseLayer.hpp"
#include "layer/VkBulgeDistortionLayer.hpp"
#include "layer/VkCannyEdgeDetectionLayer.hpp"
#include "layer/VkColorAdjustmentLayer.hpp"
#include "layer/VkColorMatrixLayer.hpp"
#include "layer/VkColourFASTFeatureDetector.hpp"
#include "layer/VkCropLayer.hpp"
#include "layer/VkDrawLayer.hpp"
#include "layer/VkEdgeDetectionLayer.hpp"
#include "layer/VkGuidedLayer.hpp"
#include "layer/VkHarrisCornerDetectionLayer.hpp"
#include "layer/VkHistogramLayer.hpp"
#include "layer/VkIOSBlurLayer.hpp"
#include "layer/VkImageProcessing.hpp"
#include "layer/VkLaplacianLayer.hpp"
#include "layer/VkLinearFilterLayer.hpp"
#include "layer/VkLookupLayer.hpp"
#include "layer/VkLowPassLayer.hpp"
#include "layer/VkLuminanceLayer.hpp"
#include "layer/VkMedianLayer.hpp"
#include "layer/VkMorphLayer.hpp"
#include "layer/VkMotionBlurLayer.hpp"
#include "layer/VkPerlinNoiseLayer.hpp"
#include "layer/VkPixellateLayer.hpp"
#include "layer/VkPoissonBlendLayer.hpp"
#include "layer/VkReduceLayer.hpp"
#include "layer/VkSeparableLinearLayer.hpp"
#include "layer/VkSphereLayer.hpp"
#include "layer/VkToonLayer.hpp"
#include "layer/VkVisualEffectLayer.hpp"
#include "layer/VkVoronoiLayer.hpp"

using namespace aoce::vulkan::layer;

namespace aoce {

ITLayer<KernelSizeParamet>* createBoxFilterLayer(ImageType imageType) {
    VkBoxBlurSLayer* boxBlur = new VkBoxBlurSLayer(imageType);
    return boxBlur;
}

ITLayer<GaussianBlurParamet>* createGaussianBlurLayer(ImageType imageType) {
    VkGaussianBlurSLayer* layer = new VkGaussianBlurSLayer(imageType);
    return layer;
}

ITLayer<ChromaKeyParamet>* createChromaKeyLayer() {
    VkChromaKeyLayer* chromKeyLayer = new VkChromaKeyLayer();
    return chromKeyLayer;
}

ITLayer<AdaptiveThresholdParamet>* createAdaptiveThresholdLayer() {
    VkAdaptiveThresholdLayer* layer = new VkAdaptiveThresholdLayer();
    return layer;
}

ITLayer<GuidedParamet>* createGuidedLayer() {
    VkGuidedLayer* layer = new VkGuidedLayer();
    return layer;
}

ITLayer<SizeScaleParamet>* createSizeScaleLayer(ImageType imageType) {
    VkSizeScaleLayer* layer = new VkSizeScaleLayer(imageType);
    return layer;
}

ITLayer<ReSizeParamet>* createResizeLayer(ImageType imageType) {
    VkResizeLayer* layer = new VkResizeLayer(imageType);
    return layer;
}

ITLayer<HarrisCornerDetectionParamet>* createHarrisCornerDetectionLayer() {
    VkHarrisCornerDetectionLayer* layer = new VkHarrisCornerDetectionLayer();
    return layer;
}

ITLayer<float>* createAverageLuminanceThresholdLayer() {
    VkAverageLuminanceThresholdLayer* layer =
        new VkAverageLuminanceThresholdLayer();
    return layer;
}

ITLayer<float>* createLuminanceThresholdLayer() {
    VkLuminanceThresholdLayer* layer = new VkLuminanceThresholdLayer();
    return layer;
}

ITLayer<BilateralParamet>* createBilateralLayer() {
    VkBilateralLayer* layer = new VkBilateralLayer();
    return layer;
}

IBaseLayer* createAddBlendLayer() {
    VkAddBlendLayer* layer = new VkAddBlendLayer();
    return layer;
}

ITLayer<float>* createAlphaBlendLayer() {
    VkAlphaBlendLayer* layer = new VkAlphaBlendLayer();
    return layer;
}

ILookupLayer* createLookupLayer() {
    VkLookupLayer* layer = new VkLookupLayer();
    return layer;
}

ITLayer<float>* createBrightnessLayer() {
    VkBrightnessLayer* layer = new VkBrightnessLayer();
    return layer;
}

ITLayer<DistortionParamet>* createBulgeDistortionLayer() {
    VkBulgeDistortionLayer* layer = new VkBulgeDistortionLayer();
    return layer;
}

ITLayer<CannyEdgeDetectionParamet>* createCannyEdgeDetectionLayer() {
    VkCannyEdgeDetectionLayer* layer = new VkCannyEdgeDetectionLayer();
    return layer;
}

IBaseLayer* createCGAColorspaceLayer() {
    VkCGAColorspaceLayer* layer = new VkCGAColorspaceLayer();
    return layer;
}

ITLayer<int32_t>* createDilationLayer(bool bSingle) {
    VkDilationLayer* layer = new VkDilationLayer(bSingle);
    return layer;
}

ITLayer<int32_t>* createErosionLayer(bool bSingle) {
    VkErosionLayer* layer = new VkErosionLayer(bSingle);
    return layer;
}

ITLayer<int32_t>* createClosingLayer(bool bSingle) {
    VkClosingLayer* layer = new VkClosingLayer(bSingle);
    return layer;
}

ITLayer<int32_t>* createOpeningLayer(bool bSingle) {
    VkOpeningLayer* layer = new VkOpeningLayer(bSingle);
    return layer;
}

IBaseLayer* createColorBlendLayer() {
    VkColorBlendLayer* layer = new VkColorBlendLayer();
    return layer;
}

IBaseLayer* createColorBurnBlendLayer() {
    VkColorBurnBlendLayer* layer = new VkColorBurnBlendLayer();
    return layer;
}

IBaseLayer* createColorDodgeBlendLayer() {
    VkColorDodgeBlendLayer* layer = new VkColorDodgeBlendLayer();
    return layer;
}

IBaseLayer* createColorInvertLayer() {
    VkColorInvertLayer* layer = new VkColorInvertLayer();
    return layer;
}

IBaseLayer* createColorLBPLayer() {
    VkColorLBPLayer* layer = new VkColorLBPLayer();
    return layer;
}

ITLayer<float>* createContrastLayer() {
    VkContrastLayer* layer = new VkContrastLayer();
    return layer;
}

ITLayer<CrosshatchParamet>* createCrosshatchLayer() {
    VkCrosshatchLayer* layer = new VkCrosshatchLayer();
    return layer;
}

ITLayer<ColorMatrixParamet>* createColorMatrixLayer() {
    VkColorMatrixLayer* layer = new VkColorMatrixLayer();
    return layer;
}

ITLayer<FASTFeatureParamet>* createColourFASTFeatureDetector() {
    VkColourFASTFeatureDetector* layer = new VkColourFASTFeatureDetector();
    return layer;
}

ITLayer<CropParamet>* createCropLayer() {
    VkCropLayer* layer = new VkCropLayer();
    return layer;
}

ITLayer<BlurPositionParamet>* createBlurPositionLayer() {
    VkGaussianBlurPositionLayer* layer = new VkGaussianBlurPositionLayer();
    return layer;
}

ITLayer<BlurSelectiveParamet>* createBlurSelectiveLayer() {
    VkGaussianBlurSelectiveLayer* layer = new VkGaussianBlurSelectiveLayer();
    return layer;
}

IBaseLayer* createDarkenBlendLayer() {
    VkDarkenBlendLayer* layer = new VkDarkenBlendLayer();
    return layer;
}

IBaseLayer* createDifferenceBlendLayer() {
    VkDifferenceBlendLayer* layer = new VkDifferenceBlendLayer();
    return layer;
}

ITLayer<float>* createDissolveBlendLayer() {
    VkDissolveBlendLayer* layer = new VkDissolveBlendLayer();
    return layer;
}

IBaseLayer* createDivideBlendLayer() {
    VkDivideBlendLayer* layer = new VkDivideBlendLayer();
    return layer;
}

ITLayer<float>* createEmbossLayer() {
    VkEmbossLayer* layer = new VkEmbossLayer();
    return layer;
}

IBaseLayer* createExclusionBlendLayer() {
    VkExclusionBlendLayer* layer = new VkExclusionBlendLayer();
    return layer;
}

ITLayer<float>* createExposureLayer() {
    VkExposureLayer* layer = new VkExposureLayer();
    return layer;
}

ITLayer<FalseColorParamet>* createFalseColorLayer() {
    VkFalseColorLayer* layer = new VkFalseColorLayer();
    return layer;
}

ITLayer<float>* createGammaLayer() {
    VkGammaLayer* layer = new VkGammaLayer();
    return layer;
}

ITLayer<SphereRefractionParamet>* createSphereRefractionLayer() {
    VkSphereRefractionLayer* layer = new VkSphereRefractionLayer();
    return layer;
}

ITLayer<SphereRefractionParamet>* createGlassSphereLayer() {
    VkGlassSphereLayer* layer = new VkGlassSphereLayer();
    return layer;
}

ITLayer<PixellateParamet>* createHalftoneLayer() {
    VkHalftoneLayer* layer = new VkHalftoneLayer();
    return layer;
}

ITLayer<PixellateParamet>* createPixellateLayer() {
    VkPixellateLayer* layer = new VkPixellateLayer();
    return layer;
}

IBaseLayer* createHardLightBlendLayer() {
    VkHardLightBlendLayer* layer = new VkHardLightBlendLayer();
    return layer;
}

ITLayer<HazeParamet>* createHazeLayer() {
    VkHazeLayer* layer = new VkHazeLayer();
    return layer;
}

ITLayer<float>* createLowPassLayer() {
    VkLowPassLayer* layer = new VkLowPassLayer();
    return layer;
}

ITLayer<float>* createHighPassLayer() {
    VkHighPassLayer* layer = new VkHighPassLayer();
    return layer;
}

IHSBLayer* createHSBLayer() {
    VkHSBLayer* layer = new VkHSBLayer();
    return layer;
}

IBaseLayer* createHueBlendLayer() {
    VkHueBlendLayer* layer = new VkHueBlendLayer();
    return layer;
}

ITLayer<float>* createHueLayer() {
    VkHueLayer* layer = new VkHueLayer();
    return layer;
}

ITLayer<HighlightShadowParamet>* createHighlightShadowLayer() {
    VKHighlightShadowLayer* layer = new VKHighlightShadowLayer();
    return layer;
}

ITLayer<HighlightShadowTintParamet>* createHighlightShadowTintLayer() {
    VKHighlightShadowTintLayer* layer = new VKHighlightShadowTintLayer();
    return layer;
}

IBaseLayer* createHistogramLayer(bool bSingle) {
    if (bSingle) {
        return new VkHistogramLayer(true);
    }
    return new VkHistogramC4Layer();
}

ITLayer<IOSBlurParamet>* createIOSBlurLayer() {
    VkIOSBlurLayer* iosBlurLayer = new VkIOSBlurLayer();
    return iosBlurLayer;
}

ITLayer<uint32_t>* createKuwaharaLayer() {
    VkKuwaharaLayer* layer = new VkKuwaharaLayer();
    return layer;
}

IBaseLayer* createLaplacianLayer(bool bsamll) {
    VkLaplacianLayer* layer = new VkLaplacianLayer(bsamll);
    return layer;
}

ITLayer<LevelsParamet>* createLevelsLayer() {
    VkLevelsLayer* layer = new VkLevelsLayer();
    return layer;
}

IBaseLayer* createLightenBlendLayer() {
    VkLightenBlendLayer* layer = new VkLightenBlendLayer();
    return layer;
}

IBaseLayer* createLinearBurnBlendLayer() {
    VkLinearBurnBlendLayer* layer = new VkLinearBurnBlendLayer();
    return layer;
}

IBaseLayer* createLuminosityBlendLayer() {
    VkLuminosityBlendLayer* layer = new VkLuminosityBlendLayer();
    return layer;
}

IBaseLayer* createLuminanceLayer() {
    VkLuminanceLayer* layer = new VkLuminanceLayer();
    return layer;
}

IBaseLayer* createMaskLayer() {
    VkMaskLayer* layer = new VkMaskLayer();
    return layer;
}

ITLayer<uint32_t>* createMedianLayer(bool bSingle) {
    VkMedianLayer* layer = new VkMedianLayer(bSingle);
    return layer;
}

IBaseLayer* createMedianK3Layer(bool bSingle) {
    VkMedianK3Layer* layer = new VkMedianK3Layer(bSingle);
    return layer;
}

ITLayer<MonochromeParamet>* createMonochromeLayer() {
    VkMonochromeLayer* layer = new VkMonochromeLayer();
    return layer;
}

ITLayer<MotionBlurParamet>* createMotionBlurLayer() {
    VkMotionBlurLayer* layer = new VkMotionBlurLayer();
    return layer;
}

IMotionDetectorLayer* createMotionDetectorLayer() {
    VkMotionDetectorLayer* layer = new VkMotionDetectorLayer();
    return layer;
}

IBaseLayer* createMultiplyBlendLayer() {
    VkMultiplyBlendLayer* layer = new VkMultiplyBlendLayer();
    return layer;
}

ITLayer<NobleCornerDetectionParamet>* createNobleCornerDetectionLayer() {
    VkNobleCornerDetectionLayer* layer = new VkNobleCornerDetectionLayer();
    return layer;
}

IBaseLayer* createNormalBlendLayer() {
    VkNormalBlendLayer* layer = new VkNormalBlendLayer();
    return layer;
}

ITLayer<float>* createOpacityLayer() {
    VkOpacityLayer* layer = new VkOpacityLayer();
    return layer;
}

IBaseLayer* createOverlayBlendLayer() {
    VkOverlayBlendLayer* layer = new VkOverlayBlendLayer();
    return layer;
}

IPerlinNoiseLayer* createPerlinNoiseLayer() {
    VkPerlinNoiseLayer* layer = new VkPerlinNoiseLayer();
    return layer;
}

ITLayer<PoissonParamet>* createPoissonBlendLayer() {
    VkPoissonBlendLayer* layer = new VkPoissonBlendLayer();
    return layer;
}

ITLayer<DistortionParamet>* createPinchDistortionLayer() {
    VkPinchDistortionLayer* layer = new VkPinchDistortionLayer();
    return layer;
}

ITLayer<SelectiveParamet>* createPixellatePositionLayer() {
    VkPixellatePositionLayer* layer = new VkPixellatePositionLayer();
    return layer;
}

ITLayer<PolarPixellateParamet>* createPolarPixellateLayer() {
    VkPolarPixellateLayer* layer = new VkPolarPixellateLayer();
    return layer;
}

ITLayer<PolkaDotParamet>* createPolkaDotLayer() {
    VkPolkaDotLayer* layer = new VkPolkaDotLayer();
    return layer;
}

ITLayer<uint32_t>* createPosterizeLayer() {
    VkPosterizeLayer* layer = new VkPosterizeLayer();
    return layer;
}

ITLayer<float>* createPrewittEdgeDetectionLayer() {
    VkPrewittEdgeDetectionLayer* layer = new VkPrewittEdgeDetectionLayer();
    return layer;
}

ITLayer<vec3>* createRGBLayer() {
    VkRGBLayer* layer = new VkRGBLayer();
    return layer;
}

IBaseLayer* createSaturationBlendLayer() {
    VkSaturationBlendLayer* layer = new VkSaturationBlendLayer();
    return layer;
}

ITLayer<float>* createSaturationLayer() {
    VkSaturationLayer* layer = new VkSaturationLayer();
    return layer;
}

IBaseLayer* createScreenBlendLayer() {
    VkScreenBlendLayer* layer = new VkScreenBlendLayer();
    return layer;
}

ITLayer<float>* createSepiaLayer() {
    VkSepiaLayer* layer = new VkSepiaLayer();
    return layer;
}

ITLayer<SharpenParamet>* createSharpenLayer() {
    VkSharpenLayer* layer = new VkSharpenLayer();
    return layer;
}

ITLayer<NobleCornerDetectionParamet>* createShiTomasiFeatureDetectionLayer() {
    VkShiTomasiFeatureDetectionLayer* layer =
        new VkShiTomasiFeatureDetectionLayer();
    return layer;
}

ITLayer<float>* createSketchLayer() {
    VkSketchLayer* layer = new VkSketchLayer();
    return layer;
}

ITLayer<SkinToneParamet>* createSkinToneLayer() {
    VkSkinToneLayer* layer = new VkSkinToneLayer();
    return layer;
}

ITLayer<SmoothToonParamet>* createSmoothToonLayer() {
    VkSmoothToonLayer* layer = new VkSmoothToonLayer();
    return layer;
}

ITLayer<float>* createSobelEdgeDetectionLayer() {
    VkSobelEdgeDetectionLayer* layer = new VkSobelEdgeDetectionLayer();
    return layer;
}

ISoftEleganceLayer* createSoftEleganceLayer() {
    VkSoftEleganceLayer* layer = new VkSoftEleganceLayer();
    return layer;
}

IBaseLayer* createSoftLightBlendLayer() {
    VkSoftLightBlendLayer* layer = new VkSoftLightBlendLayer();
    return layer;
}

ITLayer<float>* createSolarizeLayer() {
    VkSolarizeLayer* layer = new VkSolarizeLayer();
    return layer;
}

IBaseLayer* createSourceOverBlendLayer() {
    VkSourceOverBlendLayer* layer = new VkSourceOverBlendLayer();
    return layer;
}

ITLayer<vec2>* createStretchDistortionLayer() {
    VkStrectchDistortionLayer* layer = new VkStrectchDistortionLayer();
    return layer;
}

IBaseLayer* createSubtractBlendLayer() {
    VkSubtractBlendLayer* layer = new VkSubtractBlendLayer();
    return layer;
}

ITLayer<SwirlParamet>* createSwirlLayer() {
    VkSwirlLayer* layer = new VkSwirlLayer();
    return layer;
}

ITLayer<ThresholdSobelParamet>* createThresholdEdgeDetectionLayer(
    bool bSingle) {
    VkThresholdEdgeDetectionLayer* layer =
        new VkThresholdEdgeDetectionLayer(bSingle);
    return layer;
}

ITLayer<ThresholdSobelParamet>* createThresholdSketchLayer(bool bSingle) {
    VkThresholdSketchLayer* layer = new VkThresholdSketchLayer(bSingle);
    return layer;
}

ITLayer<TiltShiftParamet>* createTiltShiftLayer() {
    VkTiltShiftLayer* layer = new VkTiltShiftLayer();
    return layer;
}

ITLayer<Mat3x3>* create3x3ConvolutionLayer() {
    Vk3x3ConvolutionLayer* layer = new Vk3x3ConvolutionLayer();
    return layer;
}

ITLayer<ToonParamet>* createToonLayer() {
    VkToonLayer* layer = new VkToonLayer();
    return layer;
}

ITLayer<UnsharpMaskParamet>* createUnsharpMaskLayer() {
    VkUnsharpMaskLayer* layer = new VkUnsharpMaskLayer();
    return layer;
}

ITLayer<float>* createVibranceLayer() {
    VkVibranceLayer* layer = new VkVibranceLayer();
    return layer;
}

ITLayer<VignetteParamet>* createVignetteLayer() {
    VkVignetteLayer* layer = new VkVignetteLayer();
    return layer;
}

IBaseLayer* createVoronoiConsumerLayer() {
    VkVoronoiConsumerLayer* layer = new VkVoronoiConsumerLayer();
    return layer;
}

ITLayer<WhiteBalanceParamet>* createBalanceLayer() {
    VkWhiteBalanceLayer* layer = new VkWhiteBalanceLayer();
    return layer;
}

ITLayer<ZoomBlurParamet>* createZoomBlurLayer() {
    VkZoomBlurLayer* layer = new VkZoomBlurLayer();
    return layer;
}

IBaseLayer* createEqualizeHistLayer(bool bSingle) {
    VkEqualizeHistLayer* layer = new VkEqualizeHistLayer(bSingle);
    return layer;
}

IDrawPointsLayer* createDrawPointsLayer() {
    VkDrawPointsLayer* layer = new VkDrawPointsLayer();
    return layer;
}

IDrawRectLayer* createDrawRectLayer() {
    VkDrawRectLayer* layer = new VkDrawRectLayer();
    return layer;
}

IBaseLayer* createAlphaShowLayer() {
    VkAlphaShowLayer* layer = new VkAlphaShowLayer();
    return layer;
}

IBaseLayer* createAlphaShow2Layer() {
    VkAlphaShow2Layer* layer = new VkAlphaShow2Layer();
    return layer;
}

IBaseLayer* createConvertImageLayer() {
    VkConvertImageLayer* layer = new VkConvertImageLayer();
    return layer;
}

IBaseLayer* createAlphaSeparateLayer() {
    VkAlphaSeparateLayer* layer = new VkAlphaSeparateLayer();
    return layer;
}

IBaseLayer* createAlphaCombinLayer() {
    VkAlphaCombinLayer* layer = new VkAlphaCombinLayer();
    return layer;
}

IBaseLayer* createAlphaScaleCombinLayer() {
    VkAlphaScaleCombinLayer* layer = new VkAlphaScaleCombinLayer();
    return layer;
}

AFloatLayer* createTwoShowLayer(bool bRow) {
    VkTwoShowLayer* layer = new VkTwoShowLayer(bRow);
    return layer;
}

}  // namespace aoce