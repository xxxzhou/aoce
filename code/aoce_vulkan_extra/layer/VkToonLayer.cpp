#include "VkToonLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkToonLayer::VkToonLayer(/* args */) {
    glslPath = "glsl/toon.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkToonLayer::~VkToonLayer() {}

VkSmoothToonLayer::VkSmoothToonLayer(/* args */) {
    blurLayer = std::make_unique<VkGaussianBlurSLayer>();
    toonLayer = std::make_unique<VkToonLayer>();

    onUpdateParamet();
}

VkSmoothToonLayer::~VkSmoothToonLayer() {}

void VkSmoothToonLayer::onUpdateParamet() {
    blurLayer->updateParamet(paramet.blur);
    toonLayer->updateParamet(paramet.toon);
}

void VkSmoothToonLayer::onInitNode() {
    pipeGraph->addNode(blurLayer->getLayer())
        ->addNode(toonLayer->getLayer());
    setStartNode(blurLayer.get());
    setEndNode(toonLayer.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
