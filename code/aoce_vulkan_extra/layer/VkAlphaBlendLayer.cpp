#include "VkAlphaBlendLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAddBlendLayer::VkAddBlendLayer(/* args */) {
    glslPath = "glsl/addBlend.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkAddBlendLayer::~VkAddBlendLayer() {}

VkAlphaBlendLayer::VkAlphaBlendLayer(/* args */) {
    glslPath = "glsl/alphaBlend.comp.spv";
    inCount = 2;
    outCount = 1;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkAlphaBlendLayer::~VkAlphaBlendLayer() {}

VkHardLightBlendLayer::VkHardLightBlendLayer(/* args */) {
    glslPath = "glsl/hardLightBlend.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkHardLightBlendLayer::~VkHardLightBlendLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce