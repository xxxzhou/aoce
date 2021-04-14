#include "VkCGAColorspaceLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkCGAColorspaceLayer::VkCGAColorspaceLayer(/* args */) {
    glslPath = "glsl/cgaColorspace.comp.spv";
}

VkCGAColorspaceLayer::~VkCGAColorspaceLayer() {}

bool VkCGAColorspaceLayer::getSampled(int inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
