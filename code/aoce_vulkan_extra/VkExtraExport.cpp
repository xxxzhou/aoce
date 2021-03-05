#include "VkExtraExport.hpp"

#include "layer/VkAdaptiveThresholdLayer.hpp"
#include "layer/VkAlphaShowLayer.hpp"
#include "layer/VkChromKeyLayer.hpp"
#include "layer/VkLinearFilterLayer.hpp"
#include "layer/VkLuminanceLayer.hpp"

using namespace aoce::vulkan::layer;

namespace aoce {
namespace vulkan {

ITLayer<BoxBlueParamet>* createBoxFilterLayer() {
    VkBoxBlurLayer* boxBlur = new VkBoxBlurLayer();
    return boxBlur;
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