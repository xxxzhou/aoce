#include "VkMedianLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkMedianLayer::VkMedianLayer(bool bSingle) {
    glslPath = "glsl/median.comp.spv";
    this->bSingle = bSingle;
    if (bSingle) {
        glslPath = "glsl/medianC1.comp.spv";
    }
    setUBOSize(sizeof(paramet), true);
    paramet = 5;
    updateUBO(&paramet);
}

VkMedianLayer::~VkMedianLayer() {}

void VkMedianLayer::onInitGraph() {
    VkLayer::onInitGraph();
    if (bSingle) {
        inFormats[0].imageType = ImageType::r8;
        outFormats[0].imageType = ImageType::r8;
    }
}

VkMedianK3Layer::VkMedianK3Layer(bool bSingle) {
    this->bSingle = bSingle;
    glslPath = "glsl/medianK3.comp.spv";
    if (bSingle) {
        glslPath = "glsl/medianK3C1.comp.spv";
    }
}

VkMedianK3Layer::~VkMedianK3Layer() {}

void VkMedianK3Layer::onInitGraph() {
    VkLayer::onInitGraph();
    if (bSingle) {
        inFormats[0].imageType = ImageType::r8;
        outFormats[0].imageType = ImageType::r8;
    }
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce