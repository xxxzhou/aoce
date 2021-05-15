#include "VkColorAdjustmentLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBrightnessLayer::VkBrightnessLayer(/* args */) {
    glslPath = "glsl/brightness.comp.spv";
    paramet = 0.0f;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkBrightnessLayer::~VkBrightnessLayer() {}

VkExposureLayer::VkExposureLayer(/* args */) {
    glslPath = "glsl/exposure.comp.spv";
    paramet = 0.0f;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkExposureLayer::~VkExposureLayer() {}

VkGammaLayer::VkGammaLayer(/* args */) {
    glslPath = "glsl/gamma.comp.spv";
    paramet = 1.0f;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkGammaLayer::~VkGammaLayer() {}

VkHazeLayer::VkHazeLayer(/* args */) {
    glslPath = "glsl/haze.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkHazeLayer::~VkHazeLayer() {}

VKHighlightShadowLayer::VKHighlightShadowLayer(/* args */) {
    glslPath = "glsl/highlightShadow.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VKHighlightShadowLayer::~VKHighlightShadowLayer() {}

VKHighlightShadowTintLayer::VKHighlightShadowTintLayer(/* args */) {
    glslPath = "glsl/highlightShadowTint.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VKHighlightShadowTintLayer::~VKHighlightShadowTintLayer() {}

VkSaturationLayer::VkSaturationLayer(/* args */) {
    glslPath = "glsl/saturation.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkSaturationLayer::~VkSaturationLayer() {}

VkMonochromeLayer::VkMonochromeLayer(/* args */) {
    glslPath = "glsl/monochrome.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkMonochromeLayer::~VkMonochromeLayer() {}

VkOpacityLayer::VkOpacityLayer(/* args */) {
    glslPath = "glsl/opacity.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkOpacityLayer::~VkOpacityLayer() {}

VkRGBLayer::VkRGBLayer(/* args */) {
    glslPath = "glsl/rgb.comp.spv";
    setUBOSize(sizeof(vec3), true);
    paramet = {1.0f, 1.0f, 1.0f};
    updateUBO(&paramet);
}

VkRGBLayer::~VkRGBLayer() {}

VkSkinToneLayer::VkSkinToneLayer(/* args */) {
    glslPath = "glsl/skinTone.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkSkinToneLayer::~VkSkinToneLayer() {}

VkSolarizeLayer::VkSolarizeLayer(/* args */) {
    glslPath = "glsl/solarize.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 0.5f;
    updateUBO(&paramet);
}

VkSolarizeLayer::~VkSolarizeLayer() {}

VkVibranceLayer::VkVibranceLayer(/* args */) {
    glslPath = "glsl/vibrance.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 0.0f;
    updateUBO(&paramet);
}

VkVibranceLayer::~VkVibranceLayer() {}

VkWhiteBalanceLayer::VkWhiteBalanceLayer(/* args */) {
    glslPath = "glsl/whiteBalance.comp.spv";
    setUBOSize(sizeof(paramet));
    parametTransform();
}

VkWhiteBalanceLayer::~VkWhiteBalanceLayer() {}

void VkWhiteBalanceLayer::parametTransform() {
    // mix在值混合参数小于0后,是当负数还是当0算?
    float temperature = 0.0004 * (paramet.temperature - 5000);
    if (paramet.temperature > 5000) {
        temperature = 0.0006 * (paramet.temperature - 5000);
    }
    float tp[2] = {temperature, paramet.tint / 100.0f};
    updateUBO(tp);
}

void VkWhiteBalanceLayer::onUpdateParamet() {
    if (!(paramet == oldParamet)) {
        parametTransform();
        bParametChange = true;
    }
}

VkChromaKeyLayer::VkChromaKeyLayer(/* args */) {
    glslPath = "glsl/chromaKey.comp.spv";
    setUBOSize(sizeof(ChromaKeyParamet), true);
    updateUBO(&paramet);
}

VkChromaKeyLayer::~VkChromaKeyLayer() {}

VkColorInvertLayer::VkColorInvertLayer(/* args */) {
    glslPath = "glsl/colorInvert.comp.spv";
}

VkColorInvertLayer::~VkColorInvertLayer() {}

VkContrastLayer::VkContrastLayer(/* args */) {
    glslPath = "glsl/contrast.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkContrastLayer::~VkContrastLayer() {}

VkFalseColorLayer::VkFalseColorLayer(/* args */) {
    glslPath = "glsl/falseColor.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkFalseColorLayer::~VkFalseColorLayer() {}

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

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce