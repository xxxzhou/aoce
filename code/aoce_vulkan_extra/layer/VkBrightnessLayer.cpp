#include "VkBrightnessLayer.hpp"

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

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce