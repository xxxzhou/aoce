#include "VkLookupLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLookupLayer::VkLookupLayer(/* args */) { glslPath = "glsl/lookup.comp.spv"; }

VkLookupLayer::~VkLookupLayer() {}

bool VkLookupLayer::getSampled(int inIndex) {
    if (inIndex == 1) {
        return true;
    }
    return false;
}

bool VkLookupLayer::sampledNearest(int32_t inIndex) { return true; }

void VkLookupLayer::onInitLayer() {
    VkLayer::onInitLayer();
    if (inFormats[1].width != 512 || inFormats[1].height != 512) {
        logMessage(LogLevel::error, "look up image size must 512x512");
    }
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce