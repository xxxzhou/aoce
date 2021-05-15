#include "VkEdgeDetectionLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPrewittEdgeDetectionLayer::VkPrewittEdgeDetectionLayer(/* args */) {
    glslPath = "glsl/prewittEdge.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);
}

VkPrewittEdgeDetectionLayer::~VkPrewittEdgeDetectionLayer() {}

void VkPrewittEdgeDetectionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r32f;
}

VkSobelEdgeDetectionLayer::VkSobelEdgeDetectionLayer(/* args */) {
    glslPath = "glsl/sobelEdge.comp.spv";
}

VkSobelEdgeDetectionLayer::~VkSobelEdgeDetectionLayer() {}

VkSketchLayer::VkSketchLayer(/* args */) { glslPath = "glsl/sketch.comp.spv"; }

VkSketchLayer::~VkSketchLayer() {}

VkThresholdSketchLayer::VkThresholdSketchLayer(/* args */) {
    glslPath = "glsl/sketchThreshold.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkThresholdSketchLayer::~VkThresholdSketchLayer() {}

void VkThresholdSketchLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
}

VkThresholdEdgeDetectionLayer::VkThresholdEdgeDetectionLayer(/* args */) {
    glslPath = "glsl/sobelThreshold.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkThresholdEdgeDetectionLayer::~VkThresholdEdgeDetectionLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce