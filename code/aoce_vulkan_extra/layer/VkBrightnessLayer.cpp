#include "VkBrightnessLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBrightnessLayer::VkBrightnessLayer(/* args */) {
    glslPath = "glsl/brightness.comp.spv";
    paramet = 0.0f;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkBrightnessLayer::~VkBrightnessLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce