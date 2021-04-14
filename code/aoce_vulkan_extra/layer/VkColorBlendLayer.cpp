#include "VkColorBlendLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkColorBlendLayer::VkColorBlendLayer(/* args */) {
    glslPath = "glsl/colorBlend.comp.spv";
    inCount = 2;
}

VkColorBlendLayer::~VkColorBlendLayer() {}

VkColorBurnBlendLayer::VkColorBurnBlendLayer(/* args */) {
    glslPath = "glsl/colorBurnBlend.comp.spv";
    inCount = 2;
}

VkColorBurnBlendLayer::~VkColorBurnBlendLayer() {}

VkColorDodgeBlendLayer::VkColorDodgeBlendLayer(/* args */) {
    glslPath = "glsl/colorDodgeBlend.comp.spv";
    inCount = 2;
}

VkColorDodgeBlendLayer::~VkColorDodgeBlendLayer() {}

VkColorInvertLayer::VkColorInvertLayer(/* args */) {
    glslPath = "glsl/colorInvert.comp.spv";
}

VkColorInvertLayer::~VkColorInvertLayer() {}

VkColorLBPLayer::VkColorLBPLayer(/* args */) {
    glslPath = "glsl/colorLocalBinaryPattern.comp.spv";
}

VkColorLBPLayer::~VkColorLBPLayer() {}

VkContrastLayer::VkContrastLayer(/* args */) {
    glslPath = "glsl/contrast.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkContrastLayer::~VkContrastLayer() {}

VkCrosshatchLayer::VkCrosshatchLayer(/* args */) {
    glslPath = "glsl/crosshatch.comp.spv";
    setUBOSize(sizeof(paramet), true);  
    updateUBO(&paramet);
}

VkCrosshatchLayer::~VkCrosshatchLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce