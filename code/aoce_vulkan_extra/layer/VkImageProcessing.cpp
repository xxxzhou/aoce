#include "VkImageProcessing.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkSharpenLayer::VkSharpenLayer(/* args */) {
    glslPath = "glsl/sharpen.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkSharpenLayer::~VkSharpenLayer() {}

VkColorLBPLayer::VkColorLBPLayer(/* args */) {
    glslPath = "glsl/colorLocalBinaryPattern.comp.spv";
}

VkColorLBPLayer::~VkColorLBPLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce