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

VkExclusionBlendLayer::VkExclusionBlendLayer(/* args */) {
    glslPath = "glsl/exclusionBlend.comp.spv";
    inCount = 2;
}

VkExclusionBlendLayer::~VkExclusionBlendLayer() {}

VkFalseColorLayer::VkFalseColorLayer(/* args */) {
    glslPath = "glsl/falseColor.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkFalseColorLayer::~VkFalseColorLayer() {}

VkHueBlendLayer::VkHueBlendLayer(/* args */) {
    glslPath = "glsl/hueBlend.comp.spv";
    inCount = 2;
}

VkHueBlendLayer::~VkHueBlendLayer() {}

VkHueLayer::VkHueLayer(/* args */) {
    glslPath = "glsl/hue.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 90.0f;
    transformParamet();
}

VkHueLayer::~VkHueLayer() {}

void VkHueLayer::transformParamet() {
    float hue = fmodf(paramet, 360.0f) * M_PI / 180.0f;
    updateUBO(&hue);
}

void VkHueLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    transformParamet();
    bParametChange = true;
}

VkLevelsLayer::VkLevelsLayer(/* args */) {
    glslPath = "glsl/levels.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkLevelsLayer::~VkLevelsLayer() {}

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

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce