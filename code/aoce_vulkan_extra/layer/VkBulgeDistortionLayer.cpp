#include "VkBulgeDistortionLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBulgeDistortionLayer::VkBulgeDistortionLayer(/* args */) {
    glslPath = "glsl/bulgeDistortion.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.radius = 0.25f;
    updateUBO(&paramet);
}

VkBulgeDistortionLayer::~VkBulgeDistortionLayer() {}

bool VkBulgeDistortionLayer::getSampled(int32_t inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

VkPinchDistortionLayer::VkPinchDistortionLayer(/* args */) {
    glslPath = "glsl/pinchDistortion.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.radius = 1.0f;
    paramet.scale = 0.5f;
    updateUBO(&paramet);
}

VkPinchDistortionLayer::~VkPinchDistortionLayer() {}

bool VkPinchDistortionLayer::getSampled(int32_t inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

VkPixellatePositionLayer::VkPixellatePositionLayer(/* args */) {
    glslPath = "glsl/pixellatePosition.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.size = 0.05f;
    paramet.radius = 0.25f;
    updateUBO(&paramet);
}

VkPixellatePositionLayer::~VkPixellatePositionLayer() {}

bool VkPixellatePositionLayer::getSampled(int32_t inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

VkPolarPixellateLayer::VkPolarPixellateLayer(/* args */) {
    glslPath = "glsl/polarPixellate.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkPolarPixellateLayer::~VkPolarPixellateLayer() {}

bool VkPolarPixellateLayer::getSampled(int32_t inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

VkStrectchDistortionLayer::VkStrectchDistortionLayer(/* args */) {
    glslPath = "glsl/stretchDisortion.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.x = 0.5f;
    paramet.y = 0.5f;
    updateUBO(&paramet);
}

VkStrectchDistortionLayer::~VkStrectchDistortionLayer() {}

bool VkStrectchDistortionLayer::getSampled(int32_t inIndex) {
    return inIndex == 0;
}

VkSwirlLayer::VkSwirlLayer(/* args */) {
    glslPath = "glsl/swirl.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkSwirlLayer::~VkSwirlLayer() {}

bool VkSwirlLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce