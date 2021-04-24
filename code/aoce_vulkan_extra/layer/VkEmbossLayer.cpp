#include "VkEmbossLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkEmbossLayer::VkEmbossLayer(/* args */) {
    glslPath = "glsl/emboss.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkEmbossLayer::~VkEmbossLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce