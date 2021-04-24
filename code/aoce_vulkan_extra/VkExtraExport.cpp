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
#include "layer/VkGuidedLayer.hpp"
#include "layer/VkHarrisCornerDetectionLayer.hpp"
#include "layer/VkHistogramLayer.hpp"
#include "layer/VkLinearFilterLayer.hpp"
#include "layer/VkLookupLayer.hpp"
#include "layer/VkLowPassLayer.hpp"
#include "layer/VkLuminanceLayer.hpp"
#include "layer/VkMorphLayer.hpp"
#include "layer/VkPixellateLayer.hpp"
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

ITLayer<BulgeDistortionParamet>* createBulgeDistortionLayer() {
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

ITLayer<int>* createDilationLayer() {
    VkDilationLayer* layer = new VkDilationLayer();
    return layer;
}

ITLayer<int>* createErosionLayer() {
    VkErosionLayer* layer = new VkErosionLayer();
    return layer;
}

ITLayer<int>* createClosingLayer() {
    VkClosingLayer* layer = new VkClosingLayer();
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

ITLayer<float>* createColourFASTFeatureDetector() {
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

ITLayer<float>* createLowPassLayer() {
    VkLowPassLayer* layer = new VkLowPassLayer();
    return layer;
}

ITLayer<float>* createHighPassLayer() {
    VkHighPassLayer* layer = new VkHighPassLayer();
    return layer;
}

BaseLayer* createHistogramLayer(bool bSignal) {
    if (bSignal) {
        return new VkHistogramLayer(true);
    }
    return new VkHistogramC4Layer();
}

BaseLayer* createLuminanceLayer() {
    VkLuminanceLayer* layer = new VkLuminanceLayer();
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