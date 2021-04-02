#include "VkExtraExport.hpp"

#include "layer/VkAdaptiveThresholdLayer.hpp"
#include "layer/VkAlphaShowLayer.hpp"
#include "layer/VkBilateralLayer.hpp"
#include "layer/VkChromKeyLayer.hpp"
#include "layer/VkGuidedLayer.hpp"
#include "layer/VkHarrisCornerDetectionLayer.hpp"
#include "layer/VkLinearFilterLayer.hpp"
#include "layer/VkLuminanceLayer.hpp"
#include "layer/VkReduceLayer.hpp"
#include "layer/VkSeparableLinearLayer.hpp"

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