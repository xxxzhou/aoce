#include "VkAlphaShowLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"

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
    } else if (inFormats[0].imageType == ImageType::r32) {
        glslPath = "glsl/alphaShowSI1.comp.spv";
    } else if (inFormats[0].imageType == ImageType::rgba32) {
        glslPath = "glsl/alphaShowSI4.comp.spv";
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

VkAlphaSeparateLayer::VkAlphaSeparateLayer() {
    glslPath = "glsl/alphaSeparate.comp.spv";
    inCount = 1;
    outCount = 1;
}

VkAlphaSeparateLayer::~VkAlphaSeparateLayer() {}

void VkAlphaSeparateLayer::onInitGraph() {
    VkLayer::onInitGraph();
    //
    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
}

VkAlphaCombinLayer::VkAlphaCombinLayer() {
    glslPath = "glsl/alphaCombin.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkAlphaCombinLayer::~VkAlphaCombinLayer() {}

void VkAlphaCombinLayer::onInitGraph() {
    VkLayer::onInitGraph();
    //
    inFormats[0].imageType = ImageType::rgba8;
    inFormats[1].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::rgba8;
}

VkAlphaScaleCombinLayer::VkAlphaScaleCombinLayer() {
    glslPath = "glsl/alphaScaleCombin.comp.spv";
    inCount = 2;
    outCount = 1;
}

VkAlphaScaleCombinLayer::~VkAlphaScaleCombinLayer() {}

bool VkAlphaScaleCombinLayer::getSampled(int inIndex) { return inIndex == 1; }

VkTwoShowLayer::VkTwoShowLayer(bool bRow) {
    glslPath = "glsl/twoImageColumn.comp.spv";
    if (bRow) {
        glslPath = "glsl/twoImageRow.comp.spv";
    }
    inCount = 2;
    outCount = 1;
    paramet = 0.5f;
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkTwoShowLayer::~VkTwoShowLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce