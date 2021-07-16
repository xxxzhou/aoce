#include "VkCannyEdgeDetectionLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkDirectionalSobelEdgeDetectionLayer::VkDirectionalSobelEdgeDetectionLayer(
    /* args */) {
    glslPath = "glsl/directionalSobel.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkDirectionalSobelEdgeDetectionLayer::~VkDirectionalSobelEdgeDetectionLayer() {}

void VkDirectionalSobelEdgeDetectionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::rgba32f;
}

VkDirectionalNMS::VkDirectionalNMS(/* args */) {
    glslPath = "glsl/directionalNMS.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet.minThreshold = 0.1f;
    paramet.maxThreshold = 0.4f;
    updateUBO(&paramet);
}

VkDirectionalNMS::~VkDirectionalNMS() {}

bool VkDirectionalNMS::getSampled(int inIndex) {
    if (inIndex == 0) {
        return true;
    }
    return false;
}

void VkDirectionalNMS::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::rgba32f;
    outFormats[0].imageType = ImageType::r32f;
}

VkCannyEdgeDetectionLayer::VkCannyEdgeDetectionLayer(/* args */) {
    glslPath = "glsl/canny.comp.spv";

    luminanceLayer = std::make_unique<VkLuminanceLayer>();
    gaussianBlurLayer = std::make_unique<VkGaussianBlurSLayer>(ImageType::r8);
    sobelEDLayer = std::make_unique<VkDirectionalSobelEdgeDetectionLayer>();
    directNMSLayer = std::make_unique<VkDirectionalNMS>();

    gaussianBlurLayer->updateParamet(paramet.blueParamet);
    directNMSLayer->updateParamet({paramet.minThreshold, paramet.maxThreshold});
}

VkCannyEdgeDetectionLayer::~VkCannyEdgeDetectionLayer() {}

void VkCannyEdgeDetectionLayer::onUpdateParamet() {
    if (!(paramet.blueParamet == oldParamet.blueParamet)) {
        gaussianBlurLayer->updateParamet(paramet.blueParamet);
    }
    if (paramet.minThreshold != oldParamet.minThreshold ||
        paramet.maxThreshold != oldParamet.maxThreshold) {
        directNMSLayer->updateParamet(
            {paramet.minThreshold, paramet.maxThreshold});
    }
}

void VkCannyEdgeDetectionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r32f;
    outFormats[0].imageType = ImageType::r8;

    pipeGraph->addNode(luminanceLayer.get())
        ->addNode(gaussianBlurLayer->getLayer())
        ->addNode(sobelEDLayer->getLayer())
        ->addNode(directNMSLayer->getLayer());
}
void VkCannyEdgeDetectionLayer::onInitNode() {
    directNMSLayer->addLine(this, 0, 0);
    setStartNode(luminanceLayer.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce