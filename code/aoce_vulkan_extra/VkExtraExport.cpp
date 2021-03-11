#include "VkExtraExport.hpp"

#include "layer/VkAdaptiveThresholdLayer.hpp"
#include "layer/VkAlphaShowLayer.hpp"
#include "layer/VkChromKeyLayer.hpp"
#include "layer/VkLinearFilterLayer.hpp"
#include "layer/VkLuminanceLayer.hpp"
#include "layer/VkSeparableLinearLayer.hpp"

using namespace aoce::vulkan::layer;

namespace aoce {
namespace vulkan {

ITLayer<KernelSizeParamet>* createBoxFilterLayer() {
    VkBoxBlurSLayer* boxBlur = new VkBoxBlurSLayer();
    return boxBlur;
}

ITLayer<GaussianBlurParamet>* createGaussianBlurLayer() {
    VkGaussianBlurSLayer* layer = new VkGaussianBlurSLayer();
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

BaseLayer* createLuminanceLayer() {
    VkLuminanceLayer* layer = new VkLuminanceLayer();
    return layer;
}

BaseLayer* createAlphaShowLayer() {
    VkAlphaShowLayer* layer = new VkAlphaShowLayer();
    return layer;
}

}  // namespace vulkan
}  // namespace aoce