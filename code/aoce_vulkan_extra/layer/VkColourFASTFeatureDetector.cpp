#include "VkColourFASTFeatureDetector.hpp"

#include "aoce/Layer/PipeGraph.hpp"
#include "aoce/Layer/PipeNode.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkColourFASTFeatureDetector::VkColourFASTFeatureDetector(/* args */) {
    glslPath = "glsl/fastFeatureDetector.comp.spv";
    setUBOSize(sizeof(float));
    updateUBO(&paramet.offset);
    inCount = 2;

    boxBlur = std::make_unique<VkBoxBlurSLayer>();
    boxBlur->updateParamet({paramet.boxSize, paramet.boxSize});
}

VkColourFASTFeatureDetector::~VkColourFASTFeatureDetector() {}

bool VkColourFASTFeatureDetector::getSampled(int inIndex) {
    return inIndex == 0;
}

void VkColourFASTFeatureDetector::onUpdateParamet() {
    if (paramet.offset != oldParamet.offset) {
        updateUBO(&paramet.offset);
        bParametChange = true;
    }
    if (paramet.boxSize != oldParamet.boxSize) {
        boxBlur->updateParamet({paramet.boxSize, paramet.boxSize});
    }
}

void VkColourFASTFeatureDetector::onInitGraph() {
    VkLayer::onInitGraph();
    pipeGraph->addNode(boxBlur->getLayer());
}
void VkColourFASTFeatureDetector::onInitNode() {
    boxBlur->getNode()->addLine(getNode(), 0, 1);
    getNode()->setStartNode(getNode(), 0, 0);
    getNode()->setStartNode(boxBlur->getNode(), 0, 0);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
