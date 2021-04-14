#include "VkColorMatrixLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkColorMatrixLayer::VkColorMatrixLayer(/* args */) {
    glslPath = "glsl/colorMatrix.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkColorMatrixLayer::~VkColorMatrixLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce