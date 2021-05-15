#include "VkVisualEffectLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPosterizeLayer::VkPosterizeLayer(/* args */) {
    glslPath = "glsl/posterize.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 10;
    updateUBO(&paramet);
}

VkPosterizeLayer::~VkPosterizeLayer() {}

VkVignetteLayer::VkVignetteLayer(/* args */) {
    glslPath = "glsl/vignette.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkVignetteLayer::~VkVignetteLayer() {}

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

VkCrosshatchLayer::VkCrosshatchLayer(/* args */) {
    glslPath = "glsl/crosshatch.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkCrosshatchLayer::~VkCrosshatchLayer() {}

VkEmbossLayer::VkEmbossLayer(/* args */) {
    glslPath = "glsl/emboss.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkEmbossLayer::~VkEmbossLayer() {}

VkKuwaharaLayer::VkKuwaharaLayer(/* args */) {
    glslPath = "glsl/median.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 5;
    updateUBO(&paramet);
}

VkKuwaharaLayer::~VkKuwaharaLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce