#include "VkExtraExport.hpp"

#include "layer/VkAdaptiveThresholdLayer.hpp"
#include "layer/VkAlphaBlendLayer.hpp"
#include "layer/VkAlphaShowLayer.hpp"
#include "layer/VkBilateralLayer.hpp"
#include "layer/VkBrightnessLayer.hpp"
#include "layer/VkBulgeDistortionLayer.hpp"
#include "layer/VkCGAColorspaceLayer.hpp"
#include "layer/VkCannyEdgeDetectionLayer.hpp"
#include "layer/VkChromKeyLayer.hpp"
#include "layer/VkColorBlendLayer.hpp"
#include "layer/VkColorMatrixLayer.hpp"
#include "layer/VkColourFASTFeatureDetector.hpp"
#include "layer/VkCropLayer.hpp"
#include "layer/VkEdgeDetectionLayer.hpp"
#include "layer/VkEmbossLayer.hpp"
#include "layer/VkGuidedLayer.hpp"
#include "layer/VkHarrisCornerDetectionLayer.hpp"
#include "layer/VkHistogramLayer.hpp"
#include "layer/VkIOSBlurLayer.hpp"
#include "layer/VkKuwaharaLayer.hpp"
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

using namespace aoce::vulkan::layer;

namespace aoce {
namespace vulkan {

ITLayer<KernelSizeParamet>* createBoxFilterLayer(ImageType imageType) {
    VkBoxBlurSLayer* boxBlur = new VkBoxBlurSLayer(imageType);
    return boxBlur;
}

ITLayer<GaussianBlurParamet>* createGaussianBlurLayer(ImageType imageType) {
    VkGaussianBlurSLayer* layer = new VkGaussianBlurSLayer(imageType);
    return layer;
}

ITLayer<ChromKeyParamet>* createChromKeyLayer() {
    VkChromKeyLayer* chromKeyLayer = new VkChromKeyLayer();
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

ITLayer<BilateralParamet>* createBilateralLayer() {
    VkBilateralLayer* layer = new VkBilateralLayer();
    return layer;
}

BaseLayer* createAddBlendLayer() {
    VkAddBlendLayer* layer = new VkAddBlendLayer();
    return layer;
}

ITLayer<float>* createAlphaBlendLayer() {
    VkAlphaBlendLayer* layer = new VkAlphaBlendLayer();
    return layer;
}

LookupLayer* createLookupLayer() {
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

BaseLayer* createCGAColorspaceLayer() {
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

BaseLayer* createColorBlendLayer() {
    VkColorBlendLayer* layer = new VkColorBlendLayer();
    return layer;
}

BaseLayer* createColorBurnBlendLayer() {
    VkColorBurnBlendLayer* layer = new VkColorBurnBlendLayer();
    return layer;
}

BaseLayer* createColorDodgeBlendLayer() {
    VkColorDodgeBlendLayer* layer = new VkColorDodgeBlendLayer();
    return layer;
}

BaseLayer* createColorInvertLayer() {
    VkColorInvertLayer* layer = new VkColorInvertLayer();
    return layer;
}

BaseLayer* createColorLBPLayer() {
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

ITLayer<BulrPositionParamet>* createBlurPositionLayer() {
    VkGaussianBlurPositionLayer* layer = new VkGaussianBlurPositionLayer();
    return layer;
}

ITLayer<BlurSelectiveParamet>* createBlurSelectiveLayer() {
    VkGaussianBlurSelectiveLayer* layer = new VkGaussianBlurSelectiveLayer();
    return layer;
}

BaseLayer* createDarkenBlendLayer() {
    VkDarkenBlendLayer* layer = new VkDarkenBlendLayer();
    return layer;
}

BaseLayer* createDifferenceBlendLayer() {
    VkDifferenceBlendLayer* layer = new VkDifferenceBlendLayer();
    return layer;
}

ITLayer<float>* createDissolveBlendLayer() {
    VkDissolveBlendLayer* layer = new VkDissolveBlendLayer();
    return layer;
}

BaseLayer* createDivideBlendLayer() {
    VkDivideBlendLayer* layer = new VkDivideBlendLayer();
    return layer;
}

ITLayer<float>* createEmbossLayer() {
    VkEmbossLayer* layer = new VkEmbossLayer();
    return layer;
}

BaseLayer* createExclusionBlendLayer() {
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

BaseLayer* createHardLightBlendLayer() {
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

HSBLayer* createHSBLayer() {
    VkHSBLayer* layer = new VkHSBLayer();
    return layer;
}

BaseLayer* createHueBlendLayer() {
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

BaseLayer* createHistogramLayer(bool bSingle) {
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

BaseLayer* createLaplacianLayer(bool bsamll) {
    VkLaplacianLayer* layer = new VkLaplacianLayer(bsamll);
    return layer;
}

ITLayer<LevelsParamet>* createLevelsLayer() {
    VkLevelsLayer* layer = new VkLevelsLayer();
    return layer;
}

BaseLayer* createLightenBlendLayer() {
    VkLightenBlendLayer* layer = new VkLightenBlendLayer();
    return layer;
}

BaseLayer* createLinearBurnBlendLayer() {
    VkLinearBurnBlendLayer* layer = new VkLinearBurnBlendLayer();
    return layer;
}

BaseLayer* createLuminosityBlendLayer() {
    VkLuminosityBlendLayer* layer = new VkLuminosityBlendLayer();
    return layer;
}

BaseLayer* createLuminanceLayer() {
    VkLuminanceLayer* layer = new VkLuminanceLayer();
    return layer;
}

BaseLayer* createMaskLayer() {
    VkMaskLayer* layer = new VkMaskLayer();
    return layer;
}

ITLayer<uint32_t>* createMedianLayer(bool bSingle) {
    VkMedianLayer* layer = new VkMedianLayer(bSingle);
    return layer;
}

BaseLayer* createMedianK3Layer(bool bSingle) {
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

MotionDetectorLayer* createMotionDetectorLayer() {
    VkMotionDetectorLayer* layer = new VkMotionDetectorLayer();
    return layer;
}

BaseLayer* createMultiplyBlendLayer() {
    VkMultiplyBlendLayer* layer = new VkMultiplyBlendLayer();
    return layer;
}

ITLayer<NobleCornerDetectionParamet>* createNobleCornerDetectionLayer() {
    VkNobleCornerDetectionLayer* layer = new VkNobleCornerDetectionLayer();
    return layer;
}

BaseLayer* createNormalBlendLayer() {
    VkNormalBlendLayer* layer = new VkNormalBlendLayer();
    return layer;
}

ITLayer<float>* createOpacityLayer() {
    VkOpacityLayer* layer = new VkOpacityLayer();
    return layer;
}

BaseLayer* createOverlayBlendLayer() {
    VkOverlayBlendLayer* layer = new VkOverlayBlendLayer();
    return layer;
}

PerlinNoiseLayer* createPerlinNoiseLayer() {
    VkPerlinNoiseLayer* layer = new VkPerlinNoiseLayer();
    return layer;
}

ITLayer<PoissonParamet>* createPoissonBlendLayer() {
    VkPoissonBlendLayer* layer = new VkPoissonBlendLayer();
    return layer;
}

ITLayer<DistortionParamet>* createPinchDistorionLayer() {
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

BaseLayer* createAlphaShowLayer() {
    VkAlphaShowLayer* layer = new VkAlphaShowLayer();
    return layer;
}

BaseLayer* createAlphaShow2Layer() {
    VkAlphaShow2Layer* layer = new VkAlphaShow2Layer();
    return layer;
}

BaseLayer* createConvertImageLayer() {
    VkConvertImageLayer* layer = new VkConvertImageLayer();
    return layer;
}

}  // namespace vulkan
}  // namespace aoce