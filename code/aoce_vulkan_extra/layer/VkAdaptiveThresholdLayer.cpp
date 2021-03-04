#include "VkAdaptiveThresholdLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"
#include "aoce/Layer/PipeNode.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkAdaptiveThresholdLayer::VkAdaptiveThresholdLayer(/* args */) {
    luminance = std::make_unique<VkLuminanceLayer>();
    boxBlur = std::make_unique<VkBoxBlurLayer>(true);
    inCount = 2;
    outCount = 1;
}

VkAdaptiveThresholdLayer::~VkAdaptiveThresholdLayer() {}

void VkAdaptiveThresholdLayer::onUpdateParamet() {
    if(!(paramet.boxBlue == oldParamet.boxBlue)){
        boxBlur->updateParamet(paramet.boxBlue);
    }
}

void VkAdaptiveThresholdLayer::onInitGraph() {
    std::string path = "glsl/adaptiveThreshold.comp.spv";
    shader->loadShaderModule(context->device, path);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
    // 输入输出
    inFormats[0].imageType = ImageType::r8;
    inFormats[1].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
    // 这几个节点添加在本节点之前
    pipeGraph->addNode(luminance.get())->addNode(boxBlur->getLayer());
}

void VkAdaptiveThresholdLayer::onInitNode() {    
    luminance->getNode()->addLine(getNode(), 0, 0);
    boxBlur->getNode()->addLine(getNode(), 0, 1);
    getNode()->setStartNode(luminance->getNode());
}

// void VkAdaptiveThresholdLayer::onInitPipe() {
//     inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
//     inTexs[1]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
//     outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
//     layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &inTexs[1]->descInfo,
//                             &outTexs[0]->descInfo, &constBuf->descInfo);
//     auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
//         layout->pipelineLayout, shader->shaderStage);
//     VK_CHECK_RESULT(vkCreateComputePipelines(
//         context->device, context->pipelineCache, 1, &computePipelineInfo,
//         nullptr, &computerPipeline));
// }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce