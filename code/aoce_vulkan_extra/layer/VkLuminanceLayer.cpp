#include "VkLuminanceLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLuminanceLayer::VkLuminanceLayer(/* args */) {
    glslPath = "glsl/luminance.comp.spv";
}

VkLuminanceLayer::~VkLuminanceLayer() {}

void VkLuminanceLayer::onInitGraph() {
    VkLayer::onInitGraph();

    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
}

// luminanceRange

VkLuminanceRangeLayer::VkLuminanceRangeLayer(/* args */) {
    glslPath = "glsl/luminanceRange.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 0.6f;
    updateUBO(&paramet);
}

VkLuminanceRangeLayer::~VkLuminanceRangeLayer() {}

VkLuminanceThresholdLayer::VkLuminanceThresholdLayer(/* args */) {
    glslPath = "glsl/luminanceThreshold.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 0.5f;
    updateUBO(&paramet);
}

VkLuminanceThresholdLayer::~VkLuminanceThresholdLayer() {}

void VkLuminanceThresholdLayer::onInitGraph() {
    VkLayer::onInitGraph();

    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce