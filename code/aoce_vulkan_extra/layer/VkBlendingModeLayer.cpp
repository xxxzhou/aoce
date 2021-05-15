#include "VkBlendingModeLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAddBlendLayer::VkAddBlendLayer(/* args */) {
    glslPath = "glsl/addBlend.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkAddBlendLayer::~VkAddBlendLayer() {}

VkAlphaBlendLayer::VkAlphaBlendLayer(/* args */) {
    glslPath = "glsl/alphaBlend.comp.spv";
    inCount = 2;
    outCount = 1;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkAlphaBlendLayer::~VkAlphaBlendLayer() {}

VkHardLightBlendLayer::VkHardLightBlendLayer(/* args */) {
    glslPath = "glsl/hardLightBlend.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkHardLightBlendLayer::~VkHardLightBlendLayer() {}

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

VkExclusionBlendLayer::VkExclusionBlendLayer(/* args */) {
    glslPath = "glsl/exclusionBlend.comp.spv";
    inCount = 2;
}

VkExclusionBlendLayer::~VkExclusionBlendLayer() {}

VkHueBlendLayer::VkHueBlendLayer(/* args */) {
    glslPath = "glsl/hueBlend.comp.spv";
    inCount = 2;
}

VkHueBlendLayer::~VkHueBlendLayer() {}

VkLightenBlendLayer::VkLightenBlendLayer(/* args */) {
    glslPath = "glsl/lightenBlend.comp.spv";
    inCount = 2;
}

VkLightenBlendLayer::~VkLightenBlendLayer() {}

VkLinearBurnBlendLayer::VkLinearBurnBlendLayer(/* args */) {
    glslPath = "glsl/linearBurnBlend.comp.spv";
    inCount = 2;
}

VkLinearBurnBlendLayer::~VkLinearBurnBlendLayer() {}

VkLuminosityBlendLayer::VkLuminosityBlendLayer(/* args */) {
    glslPath = "glsl/luminosityBlend.comp.spv";
    inCount = 2;
}

VkLuminosityBlendLayer::~VkLuminosityBlendLayer() {}

VkMaskLayer::VkMaskLayer(/* args */) {
    glslPath = "glsl/mask.comp.spv";
    inCount = 2;
}

VkMaskLayer::~VkMaskLayer() {}

VkMultiplyBlendLayer::VkMultiplyBlendLayer(/* args */) {
    glslPath = "glsl/multiplyBlend.comp.spv";
    inCount = 2;
}

VkMultiplyBlendLayer::~VkMultiplyBlendLayer() {}

VkNormalBlendLayer::VkNormalBlendLayer(/* args */) {
    glslPath = "glsl/normalBlend.comp.spv";
    inCount = 2;
}

VkNormalBlendLayer::~VkNormalBlendLayer() {}

VkOverlayBlendLayer::VkOverlayBlendLayer(/* args */) {
    glslPath = "glsl/overlayBlend.comp.spv";
    inCount = 2;
}

VkOverlayBlendLayer::~VkOverlayBlendLayer() {}

VkSaturationBlendLayer::VkSaturationBlendLayer(/* args */) {
    glslPath = "glsl/saturationBlend.comp.spv";
    inCount = 2;
}

VkSaturationBlendLayer::~VkSaturationBlendLayer() {}

VkScreenBlendLayer::VkScreenBlendLayer(/* args */) {
    glslPath = "glsl/screenBlend.comp.spv";
    inCount = 2;
}

VkScreenBlendLayer::~VkScreenBlendLayer() {}

VkSoftLightBlendLayer::VkSoftLightBlendLayer(/* args */) {
    glslPath = "glsl/softLightBlend.comp.spv";
    inCount = 2;
}

VkSoftLightBlendLayer::~VkSoftLightBlendLayer() {}

VkSourceOverBlendLayer::VkSourceOverBlendLayer(/* args */) {
    glslPath = "glsl/sourceOverBlend.comp.spv";
    inCount = 2;
}

VkSourceOverBlendLayer::~VkSourceOverBlendLayer() {}

VkSubtractBlendLayer::VkSubtractBlendLayer(/* args */) {
    glslPath = "glsl/subtractBlend.comp.spv";
    inCount = 2;
}

VkSubtractBlendLayer::~VkSubtractBlendLayer() {}

VkDarkenBlendLayer::VkDarkenBlendLayer(/* args */) {
    glslPath = "glsl/darkenBlend.comp.spv";
    inCount = 2;
}

VkDarkenBlendLayer::~VkDarkenBlendLayer() {}

VkDifferenceBlendLayer::VkDifferenceBlendLayer(/* args */) {
    glslPath = "glsl/differenceBlend.comp.spv";
    inCount = 2;
}

VkDifferenceBlendLayer::~VkDifferenceBlendLayer() {}

VkDissolveBlendLayer::VkDissolveBlendLayer(/* args */) {
    glslPath = "glsl/dissolveBlend.comp.spv";
    inCount = 2;
    setUBOSize(sizeof(paramet), true);
    paramet = 0.5f;
    updateUBO(&paramet);
}

VkDissolveBlendLayer::~VkDissolveBlendLayer() {}

VkDivideBlendLayer::VkDivideBlendLayer(/* args */) {
    glslPath = "glsl/divideBlend.comp.spv";
    inCount = 2;
}

VkDivideBlendLayer::~VkDivideBlendLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce