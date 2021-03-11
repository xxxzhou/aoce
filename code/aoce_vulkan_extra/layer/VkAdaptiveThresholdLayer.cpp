#include "VkAdaptiveThresholdLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"
#include "aoce/Layer/PipeNode.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAdaptiveThresholdLayer::VkAdaptiveThresholdLayer(/* args */) {
    setUBOSize(4);
    luminance = std::make_unique<VkLuminanceLayer>();
    boxBlur = std::make_unique<VkBoxBlurLayer>(true);
    // kernel size会导致Graph重置,在onInitGraph之前更新下,避免可能的二次重置
    boxBlur->updateParamet({paramet.boxSize, paramet.boxSize});
    inCount = 2;
    outCount = 1;
    glslPath = "glsl/adaptiveThreshold.comp.spv";
}

VkAdaptiveThresholdLayer::~VkAdaptiveThresholdLayer() {}

void VkAdaptiveThresholdLayer::onUpdateParamet() {
    if (paramet.boxSize != oldParamet.boxSize) {
        boxBlur->updateParamet({paramet.boxSize, paramet.boxSize});
    }
    if (paramet.offset != oldParamet.offset) {
        memcpy(constBufCpu.data(), &paramet.offset, conBufSize);
        bParametChange = true;
    }
}

void VkAdaptiveThresholdLayer::onInitGraph() {
    VkLayer::onInitGraph();
    // 输入输出
    inFormats[0].imageType = ImageType::r8;
    inFormats[1].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
    // 这几个节点添加在本节点之前
    pipeGraph->addNode(luminance.get())->addNode(boxBlur->getLayer());
    // 更新下默认UBO信息
    memcpy(constBufCpu.data(), &paramet.offset, conBufSize);
}

void VkAdaptiveThresholdLayer::onInitNode() {
    luminance->getNode()->addLine(getNode(), 0, 0);
    boxBlur->getNode()->addLine(getNode(), 0, 1);
    getNode()->setStartNode(luminance->getNode());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce