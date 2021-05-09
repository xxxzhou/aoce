#include "VkToonLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkToonLayer::VkToonLayer(/* args */) {
    glslPath = "glsl/toon.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkToonLayer::~VkToonLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
