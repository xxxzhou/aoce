#include "VkColourFASTFeatureDetector.hpp"

#include "aoce/Layer/PipeGraph.hpp"
#include "aoce/Layer/PipeNode.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkColourFASTFeatureDetector::VkColourFASTFeatureDetector(/* args */) {
    glslPath = "glsl/fastFeatureDetector.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 1.0f;
    updateUBO(&paramet);

    // boxBlur = std::make_unique<VkBoxBlurSLayer>();
}

VkColourFASTFeatureDetector::~VkColourFASTFeatureDetector() {}

bool VkColourFASTFeatureDetector::getSampled(int inIndex) {
    return inIndex == 0;
}

// void VkColourFASTFeatureDetector::onInitGraph() {
//     VkLayer::onInitGraph();
//     pipeGraph->addNode(boxBlur->getLayer());
// }
// void VkColourFASTFeatureDetector::onInitNode() {
//     getNode()->setStartNode(boxBlur->getNode());
// }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
