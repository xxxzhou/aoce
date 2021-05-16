#include "VkAdaptiveThresholdLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAdaptiveThresholdLayer::VkAdaptiveThresholdLayer(/* args */) {
    setUBOSize(4);
    inCount = 2;
    outCount = 1;
    glslPath = "glsl/adaptiveThreshold.comp.spv";
    //
    luminance = std::make_unique<VkLuminanceLayer>();
    boxBlur = std::make_unique<VkBoxBlurSLayer>(ImageType::r8);
    // kernel size会导致Graph重置,在onInitGraph之前更新下,避免可能的二次重置
    boxBlur->updateParamet({paramet.boxSize, paramet.boxSize});
    updateUBO(&paramet.offset);
}

VkAdaptiveThresholdLayer::~VkAdaptiveThresholdLayer() {}

void VkAdaptiveThresholdLayer::onUpdateParamet() {
    if (paramet.boxSize != oldParamet.boxSize) {
        boxBlur->updateParamet({paramet.boxSize, paramet.boxSize});
    }
    if (paramet.offset != oldParamet.offset) {
        updateUBO(&paramet.offset);
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
}

void VkAdaptiveThresholdLayer::onInitNode() {
    luminance->addLine(getLayer(), 0, 0);
    boxBlur->addLine(getLayer(), 0, 1);
    setStartNode(luminance.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce