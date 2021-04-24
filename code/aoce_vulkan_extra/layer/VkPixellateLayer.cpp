#include "VkPixellateLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPixellateLayer::VkPixellateLayer(/* args */) {
    glslPath = "glsl/glassSphere.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkPixellateLayer::~VkPixellateLayer() {}

VkHalftoneLayer::VkHalftoneLayer(/* args */) {
    glslPath = "glsl/halftone.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkHalftoneLayer::~VkHalftoneLayer() {}

bool VkHalftoneLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce