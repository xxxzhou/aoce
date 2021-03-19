#include "VkAlphaShowLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAlphaShowLayer::VkAlphaShowLayer() {
    // inFormats[0].imageType由上一层决定
    bAutoImageType = true;
}

VkAlphaShowLayer::~VkAlphaShowLayer() {}

void VkAlphaShowLayer::onInitLayer() {
    glslPath = "glsl/alphaShow.comp.spv";
    if (inFormats[0].imageType == ImageType::r8) {
        glslPath = "glsl/alphaShowC1.comp.spv";
    } else if (inFormats[0].imageType == ImageType::rgbaf32) {
        glslPath = "glsl/alphaShowF4.comp.spv";
    }
    onInitGraph();
    outFormats[0].imageType = ImageType::rgba8;
    VkLayer::onInitLayer();
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce