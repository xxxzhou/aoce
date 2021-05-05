#include "VkPoissonBlendLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPoissonBlendLayer::VkPoissonBlendLayer(/* args */) {
    glslPath = "glsl/poissonBlend.comp.spv";
    inCount = 2;
}

VkPoissonBlendLayer::~VkPoissonBlendLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce