#include "VkIOSBlurLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkIOSBlurLayer::VkIOSBlurLayer(/* args */) {
    downLayer = std::make_unique<VkSizeScaleLayer>();
    saturationLayer = std::make_unique<VkSaturationLayer>();
    blurLayer = std::make_unique<VkGaussianBlurSLayer>();
    lumRangeLayer = std::make_unique<VkLuminanceRangeLayer>();
    upLayer = std::make_unique<VkSizeScaleLayer>();

    onUpdateParamet();
}

VkIOSBlurLayer::~VkIOSBlurLayer() {}

void VkIOSBlurLayer::onUpdateParamet() {
    downLayer->updateParamet({1, 1.0f / paramet.sacle, 1.0f / paramet.sacle});
    saturationLayer->updateParamet(paramet.saturation);
    blurLayer->updateParamet(paramet.blurParamet);
    lumRangeLayer->updateParamet(paramet.range);
    upLayer->updateParamet({1, paramet.sacle, paramet.sacle});
}

void VkIOSBlurLayer::onInitNode() {
    pipeGraph->addNode(downLayer->getLayer())
        ->addNode(saturationLayer->getLayer())
        ->addNode(blurLayer->getLayer())
        ->addNode(lumRangeLayer->getLayer())
        ->addNode(upLayer->getLayer());
    setStartNode(downLayer.get());
    setEndNode(upLayer.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce