#include "VkKuwaharaLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkKuwaharaLayer::VkKuwaharaLayer(/* args */) {
    glslPath = "glsl/median.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 5;
    updateUBO(&paramet);
}

VkKuwaharaLayer::~VkKuwaharaLayer() {}



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce