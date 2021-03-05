#include "VkAlphaShowLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAlphaShowLayer::VkAlphaShowLayer(/* args */) {
    glslPath = "glsl/alphaShow.comp.spv";
}

VkAlphaShowLayer::~VkAlphaShowLayer() {}

void VkAlphaShowLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::rgba8;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce