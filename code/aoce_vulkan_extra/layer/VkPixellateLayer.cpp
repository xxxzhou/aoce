#include "VkPixellateLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPixellateLayer::VkPixellateLayer(/* args */) {
    glslPath = "glsl/pixellate.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.fractionalWidthOfPixel = 0.05;
    paramet.aspectRatio = 1.0f;
    updateUBO(&paramet);
}

VkPixellateLayer::~VkPixellateLayer() {}

bool VkPixellateLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

VkHalftoneLayer::VkHalftoneLayer(/* args */) {
    glslPath = "glsl/halftone.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.fractionalWidthOfPixel = 0.01;
    paramet.aspectRatio = 1.0f;
    updateUBO(&paramet);
}

VkHalftoneLayer::~VkHalftoneLayer() {}

bool VkHalftoneLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

VkPolkaDotLayer::VkPolkaDotLayer(/* args */) {
    glslPath = "glsl/polkaDot.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.fractionalWidthOfPixel = 0.01;
    paramet.aspectRatio = 1.0f;
    paramet.dotScaling = 0.90f;
    updateUBO(&paramet);
}

VkPolkaDotLayer::~VkPolkaDotLayer() {}

bool VkPolkaDotLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce