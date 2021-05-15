#include "VkBlurBlendBaseLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBlurBlendBaseLayer::VkBlurBlendBaseLayer(/* args */) {
    inCount = 2;
    blurLayer = std::make_unique<VkGaussianBlurSLayer>();
}

VkBlurBlendBaseLayer::~VkBlurBlendBaseLayer() {}

void VkBlurBlendBaseLayer::onInitGraph() {
    VkLayer::onInitGraph();
    pipeGraph->addNode(blurLayer->getLayer());
}

void VkBlurBlendBaseLayer::onInitNode() {
    blurLayer->addLine(this, 0, 1);
    setStartNode(this);
    setStartNode(blurLayer.get());
}

void VkBlurBlendBaseLayer::baseParametChange(
    const GaussianBlurParamet& baseParamet) {
    blurLayer->updateParamet(baseParamet);
}

VkGaussianBlurPositionLayer::VkGaussianBlurPositionLayer(/* args */) {
    glslPath = "glsl/bulrPosition.comp.spv";
    setUBOSize(sizeof(paramet.bulrPosition));
    updateUBO(&paramet.bulrPosition);
    baseParametChange(paramet.gaussian);
}

VkGaussianBlurPositionLayer::~VkGaussianBlurPositionLayer() {}

void VkGaussianBlurPositionLayer::onUpdateParamet() {
    if (!(paramet.bulrPosition == oldParamet.bulrPosition)) {
        updateUBO(&paramet.bulrPosition);
        bParametChange = true;
    }
    if (!(paramet.gaussian == oldParamet.gaussian)) {
        blurLayer->updateParamet(paramet.gaussian);
    }
}

bool VkGaussianBlurPositionLayer::getSampled(int32_t inIndex) {
    return inIndex == 0 || inIndex == 1;
}

VkGaussianBlurSelectiveLayer::VkGaussianBlurSelectiveLayer(/* args */) {
    glslPath = "glsl/blurSelective.comp.spv";
    setUBOSize(sizeof(paramet.bulrPosition));
    updateUBO(&paramet.bulrPosition);
    baseParametChange(paramet.gaussian);
}

VkGaussianBlurSelectiveLayer::~VkGaussianBlurSelectiveLayer() {}

void VkGaussianBlurSelectiveLayer::onUpdateParamet() {
    if (!(paramet.bulrPosition == oldParamet.bulrPosition)) {
        updateUBO(&paramet.bulrPosition);
        bParametChange = true;
    }
    if (!(paramet.gaussian == oldParamet.gaussian)) {
        blurLayer->updateParamet(paramet.gaussian);
    }
}

bool VkGaussianBlurSelectiveLayer::getSampled(int32_t inIndex) {
    return inIndex == 0 || inIndex == 1;
}

VkTiltShiftLayer::VkTiltShiftLayer(/* args */) {
    glslPath = "glsl/tiltShift.comp.spv";
    setUBOSize(sizeof(float) * 3);
    transformParamet();
    baseParametChange(paramet.blur);
}

VkTiltShiftLayer::~VkTiltShiftLayer() {}

void VkTiltShiftLayer::transformParamet() {
    std::vector<float> paramets = {paramet.topFocusLevel,
                                   paramet.bottomFocusLevel,
                                   paramet.focusFallOffRate};
    updateUBO(paramets.data());
}

void VkTiltShiftLayer::onUpdateParamet() {
    if (!(paramet.blur == oldParamet.blur)) {
        baseParametChange(paramet.blur);
    }
    if (paramet.bottomFocusLevel != oldParamet.bottomFocusLevel ||
        paramet.focusFallOffRate != oldParamet.focusFallOffRate ||
        paramet.topFocusLevel != oldParamet.topFocusLevel) {
        transformParamet();
        bParametChange = true;
    }
}

VkUnsharpMaskLayer::VkUnsharpMaskLayer(/* args */) {
    glslPath = "glsl/unsharpMask.comp.spv";
    setUBOSize(sizeof(float));
    updateUBO(&paramet.intensity);
    baseParametChange(paramet.blur);
}

VkUnsharpMaskLayer::~VkUnsharpMaskLayer() {}

void VkUnsharpMaskLayer::onUpdateParamet() {
    if (!(paramet.blur == oldParamet.blur)) {
        baseParametChange(paramet.blur);
    }
    if (paramet.intensity != oldParamet.intensity) {
        updateUBO(&paramet.intensity);
        bParametChange = true;
    }
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce