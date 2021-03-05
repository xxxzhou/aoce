#include "VkLuminanceLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLuminanceLayer::VkLuminanceLayer(/* args */) {
    glslPath = "glsl/luminance.comp.spv";
}

VkLuminanceLayer::~VkLuminanceLayer() {}

void VkLuminanceLayer::onInitGraph() {
    VkLayer::onInitGraph();

    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce