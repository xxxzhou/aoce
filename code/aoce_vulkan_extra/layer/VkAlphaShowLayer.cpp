#include "VkAlphaShowLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAlphaShowLayer::VkAlphaShowLayer() {
    // inFormats[0].imageType由上一层决定
    bAutoImageType = true;
}

VkAlphaShowLayer::~VkAlphaShowLayer() {}

void VkAlphaShowLayer::onInitLayer() {
    glslPath = "glsl/alphaShow.comp.spv";
    if (inFormats[0].imageType == ImageType::r8) {
        glslPath = "glsl/alphaShowC1.comp.spv";
    } else if (inFormats[0].imageType == ImageType::rgba32f) {
        glslPath = "glsl/alphaShowF4.comp.spv";
    } else if (inFormats[0].imageType == ImageType::r32f) {
        glslPath = "glsl/alphaShowF1.comp.spv";
    }
    // 加载shader
    onInitGraph();
    outFormats[0].imageType = ImageType::rgba8;
    VkLayer::onInitLayer();
}

VkAlphaShow2Layer::VkAlphaShow2Layer() {
    glslPath = "glsl/showRound.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkAlphaShow2Layer::~VkAlphaShow2Layer() {}

void VkAlphaShow2Layer::onInitGraph() {
    VkLayer::onInitGraph();
    //
    inFormats[0].imageType = ImageType::r8;
    inFormats[1].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::rgba8;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce