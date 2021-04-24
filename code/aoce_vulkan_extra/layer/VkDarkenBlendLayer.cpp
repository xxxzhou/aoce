#include "VkDarkenBlendLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

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