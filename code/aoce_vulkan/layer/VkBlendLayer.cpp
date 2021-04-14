#include "VkBlendLayer.hpp"

#include <math.h>

#include "VkPipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBlendLayer::VkBlendLayer(/* args */) {
    glslPath = "glsl/resize.comp.spv";
    setUBOSize(sizeof(VkBlendParamet));
    inCount = 2;
    outCount = 1;
    parametTransform();
}

VkBlendLayer::~VkBlendLayer() {}

void VkBlendLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    parametTransform();
    // 运行时更新UBO
    bParametChange = true;
}

void VkBlendLayer::parametTransform() {
    vkParamet.centerX = paramet.centerX;
    vkParamet.centerY = paramet.centerY;
    vkParamet.width = paramet.width;
    vkParamet.height = paramet.height;
    vkParamet.opacity = std::max(0.f, std::min(1.0f, paramet.alaph));
    if (inFormats.size() > 1) {
        vkParamet.fx = inFormats[1].width / vkParamet.width;
        vkParamet.fy = inFormats[1].height / vkParamet.height;
    }
    updateUBO(&vkParamet);
}

bool VkBlendLayer::getSampled(int inIndex) { return inIndex == 1; }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
