#include "VkOperateLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkOperateLayer::VkOperateLayer(/* args */) {
    glslPath = "glsl/operate.comp.spv";
    setUBOSize(sizeof(TexOperateParamet), true);
    updateUBO(&paramet);
}

VkOperateLayer::~VkOperateLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce