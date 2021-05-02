#include "VkResizeLayer.hpp"

#include "VkPipeGraph.hpp"

#define IS_SAMPLER 1

namespace aoce {
namespace vulkan {
namespace layer {

VkResizeLayer::VkResizeLayer() : VkResizeLayer(ImageType::rgba8) {}

VkResizeLayer::VkResizeLayer(ImageType imageType) {
    this->imageType = imageType;
    setUBOSize(sizeof(SizeScaleParamet));
    glslPath = "glsl/resize.comp.spv";
    if (imageType == ImageType::r8) {
        glslPath = "glsl/resizeC1.comp.spv";
    } else if (imageType == ImageType::rgba32f) {
        glslPath = "glsl/resizeF4.comp.spv";
    }
    paramet.bLinear = true;
    paramet.newWidth = 1920;
    paramet.newHeight = 1080;
}

VkResizeLayer::~VkResizeLayer() {}

bool VkResizeLayer::getSampled(int inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

bool VkResizeLayer::sampledNearest(int32_t inIndex) {
    return paramet.bLinear == 0;
}

void VkResizeLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    resetGraph();
}

void VkResizeLayer::onInitGraph() {
    inFormats[0].imageType = imageType;
    outFormats[0].imageType = imageType;
    VkLayer::onInitGraph();
}

void VkResizeLayer::onInitLayer() {
    assert(paramet.newWidth > 0 && paramet.newHeight > 0);
    outFormats[0].width = paramet.newWidth;
    outFormats[0].height = paramet.newHeight;
    SizeScaleParamet vpar = {};
    vpar.bLinear = paramet.bLinear;
    vpar.fx = (float)inFormats[0].width / outFormats[0].width;
    vpar.fy = (float)inFormats[0].height / outFormats[0].height;
    updateUBO(&vpar);

    sizeX = divUp(outFormats[0].width, groupX);
    sizeY = divUp(outFormats[0].height, groupY);
}

VkSizeScaleLayer::VkSizeScaleLayer() : VkSizeScaleLayer(ImageType::rgba8) {}

VkSizeScaleLayer::VkSizeScaleLayer(ImageType imageType) {
    this->imageType = imageType;
    setUBOSize(sizeof(SizeScaleParamet));
    glslPath = "glsl/resize.comp.spv";
    if (imageType == ImageType::r8) {
        glslPath = "glsl/resizeC1.comp.spv";
    } else if (imageType == ImageType::rgba32f) {
        glslPath = "glsl/resizeF4.comp.spv";
    }
    paramet.bLinear = true;
    paramet.fx = 1.0f;
    paramet.fy = 1.0f;
}

VkSizeScaleLayer::~VkSizeScaleLayer() {}

bool VkSizeScaleLayer::getSampled(int inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

bool VkSizeScaleLayer::sampledNearest(int32_t inIndex) {
    return paramet.bLinear == 0;
}

void VkSizeScaleLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    resetGraph();
}

void VkSizeScaleLayer::onInitGraph() {
    inFormats[0].imageType = imageType;
    outFormats[0].imageType = imageType;
    VkLayer::onInitGraph();
}

void VkSizeScaleLayer::onInitLayer() {
    assert(paramet.fx > 0.0f && paramet.fx > 0.0f);
    outFormats[0].width = inFormats[0].width * paramet.fx;
    outFormats[0].height = inFormats[0].width * paramet.fx;
    paramet.fx = 1.0f / paramet.fx;
    paramet.fy = 1.0f / paramet.fy;
    updateUBO(&paramet);

    sizeX = divUp(outFormats[0].width, groupX);
    sizeY = divUp(outFormats[0].height, groupY);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce