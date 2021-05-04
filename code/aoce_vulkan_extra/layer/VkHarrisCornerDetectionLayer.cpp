#include "VkHarrisCornerDetectionLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkXYDerivativeLayer::VkXYDerivativeLayer() {
    glslPath = "glsl/prewitt.comp.spv";
    setUBOSize(sizeof(float), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkXYDerivativeLayer::~VkXYDerivativeLayer() {}

void VkXYDerivativeLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::rgba32f;
}

VkThresholdedNMS::VkThresholdedNMS() {
    glslPath = "glsl/thresholdedNMS.comp.spv";
    setUBOSize(sizeof(float), true);
    paramet = 0.9f;
    updateUBO(&paramet);
}

VkThresholdedNMS::~VkThresholdedNMS() {}

void VkThresholdedNMS::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r32f;
    outFormats[0].imageType = ImageType::r8;
}

VkHarrisCornerDetectionLayer::VkHarrisCornerDetectionLayer(/* args */) {
    // self
    glslPath = "glsl/harrisCornerDetection.comp.spv";
    setUBOSize(8);
    //
    xyDerivativeLayer = std::make_unique<VkXYDerivativeLayer>();
    blurLayer = std::make_unique<VkGaussianBlurSLayer>(ImageType::rgba32f);
    thresholdNMSLayer = std::make_unique<VkThresholdedNMS>();
    //
    xyDerivativeLayer->updateParamet(paramet.edgeStrength);
    blurLayer->updateParamet(paramet.blueParamet);
    thresholdNMSLayer->updateParamet(paramet.threshold);
    updateUBO();
}

VkHarrisCornerDetectionLayer::~VkHarrisCornerDetectionLayer() {}

void VkHarrisCornerDetectionLayer::updateUBO() {
    std::vector<float> paramets = {paramet.harris, paramet.sensitivity};
    VkLayer::updateUBO(paramets.data());
}

void VkHarrisCornerDetectionLayer::onUpdateParamet() {
    if (paramet.edgeStrength != oldParamet.edgeStrength) {
        xyDerivativeLayer->updateParamet(paramet.edgeStrength);
    }
    if (!(paramet.blueParamet == oldParamet.blueParamet)) {
        blurLayer->updateParamet(paramet.blueParamet);
    }
    if (paramet.threshold != oldParamet.threshold) {
        thresholdNMSLayer->updateParamet(paramet.threshold);
    }
    if (paramet.harris != oldParamet.harris ||
        paramet.sensitivity != oldParamet.sensitivity) {
        updateUBO();
        bParametChange = true;
    }
}

void VkHarrisCornerDetectionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::rgba32f;
    outFormats[0].imageType = ImageType::r32f;
    pipeGraph->addNode(xyDerivativeLayer->getLayer())
        ->addNode(blurLayer->getLayer());
    pipeGraph->addNode(thresholdNMSLayer->getLayer());
}

void VkHarrisCornerDetectionLayer::onInitNode() {
    blurLayer->addLine(this, 0, 0);
    addLine(thresholdNMSLayer.get(), 0, 0);
    setStartNode(xyDerivativeLayer.get());
    setEndNode(thresholdNMSLayer.get());
}

VkNobleCornerDetectionLayer::VkNobleCornerDetectionLayer(/* args */) {
    glslPath = "glsl/nobleCornerDetection.comp.spv";
}

VkNobleCornerDetectionLayer::~VkNobleCornerDetectionLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce