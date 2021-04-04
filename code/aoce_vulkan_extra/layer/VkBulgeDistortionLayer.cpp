#include "VkBulgeDistortionLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBulgeDistortionLayer::VkBulgeDistortionLayer(/* args */) {
    glslPath = "glsl/bulgeDistortion.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkBulgeDistortionLayer::~VkBulgeDistortionLayer() {}

bool VkBulgeDistortionLayer::getSampled(int inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce