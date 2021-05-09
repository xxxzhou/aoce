#pragma once

#include "AoceVkExtra.hpp"

namespace aoce {
namespace vulkan {

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

AOCE_VE_EXPORT ITLayer<DistortionParamet>* createBulgeDistortionLayer();

AOCE_VE_EXPORT ITLayer<CannyEdgeDetectionParamet>*
createCannyEdgeDetectionLayer();

AOCE_VE_EXPORT BaseLayer* createCGAColorspaceLayer();

AOCE_VE_EXPORT ITLayer<int32_t>* createDilationLayer(bool bSingle);

AOCE_VE_EXPORT ITLayer<int32_t>* createErosionLayer(bool bSingle);

AOCE_VE_EXPORT ITLayer<int32_t>* createClosingLayer(bool bSingle);

AOCE_VE_EXPORT ITLayer<int32_t>* createOpeningLayer(bool bSingle);

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

AOCE_VE_EXPORT ITLayer<NobleCornerDetectionParamet>*
createNobleCornerDetectionLayer();

AOCE_VE_EXPORT BaseLayer* createNormalBlendLayer();

AOCE_VE_EXPORT ITLayer<float>* createOpacityLayer();

AOCE_VE_EXPORT BaseLayer* createOverlayBlendLayer();

AOCE_VE_EXPORT PerlinNoiseLayer* createPerlinNoiseLayer();

AOCE_VE_EXPORT ITLayer<PoissonParamet>* createPoissonBlendLayer();

AOCE_VE_EXPORT ITLayer<DistortionParamet>* createPinchDistorionLayer();

AOCE_VE_EXPORT ITLayer<SelectiveParamet>* createPixellatePositionLayer();

AOCE_VE_EXPORT ITLayer<PolarPixellateParamet>* createPolarPixellateLayer();

AOCE_VE_EXPORT ITLayer<PolkaDotParamet>* createPolkaDotLayer();

AOCE_VE_EXPORT ITLayer<uint32_t>* createPosterizeLayer();

AOCE_VE_EXPORT ITLayer<float>* createPrewittEdgeDetectionLayer();

AOCE_VE_EXPORT BaseLayer* createAlphaShowLayer();

AOCE_VE_EXPORT BaseLayer* createAlphaShow2Layer();

AOCE_VE_EXPORT BaseLayer* createConvertImageLayer();

}  // namespace vulkan
}  // namespace aoce
