#include "VkLaplacianLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLaplacianLayer::VkLaplacianLayer(bool small) {
    glslPath = "glsl/filterMat3x3.comp.spv";
    setUBOSize(sizeof(Mat3x3));
    if (small) {
        mat.col0 = {0.0f, 1.0f, 0.0f};
        mat.col1 = {1.0f, -4.0f, 1.0f};
        mat.col2 = {0.0f, 1.0f, 0.0f};
    } else {
        mat.col0 = {2.0f, 0.0f, 2.0f};
        mat.col1 = {0.0f, -8.0f, 0.0f};
        mat.col2 = {2.0f, 0.0f, 2.0f};
    }
    updateUBO(&mat);
}

VkLaplacianLayer::~VkLaplacianLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce