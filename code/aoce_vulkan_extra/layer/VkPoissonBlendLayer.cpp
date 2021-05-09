#include "VkPoissonBlendLayer.hpp"

#include "aoce_vulkan/layer/VkPipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkCopyImageLayer::VkCopyImageLayer(/* args */) {
    glslPath = "glsl/copyImage.comp.spv";
}

VkCopyImageLayer::~VkCopyImageLayer() {}

VkPoissonBlendLayer::VkPoissonBlendLayer(/* args */) {
    glslPath = "glsl/poissonBlend.comp.spv";
    inCount = 2;
    setUBOSize(sizeof(float));
    updateUBO(&paramet.percent);

    copyLayer = std::make_unique<VkCopyImageLayer>();
}

VkPoissonBlendLayer::~VkPoissonBlendLayer() {}

void VkPoissonBlendLayer::onUpdateParamet() {
    if (paramet.iterationNum != oldParamet.iterationNum) {
        resetGraph();
    }
    if (paramet.percent != oldParamet.percent) {
        updateUBO(&paramet.percent);
        bParametChange = true;
    }
}

void VkPoissonBlendLayer::onInitGraph() {   
    if (!glslPath.empty()) {
        shader->loadShaderModule(context->device, glslPath);
    }
    // 定义pushConstant
    std::vector<UBOLayoutItem> items;
    items.push_back(
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT});
    items.push_back(
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT});
    items.push_back(
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT});
    items.push_back(
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT});
    layout->addSetLayout(items);
    layout->generateLayout(sizeof(int32_t));

    pipeGraph->addNode(copyLayer.get());
}

void VkPoissonBlendLayer::onInitNode() {
    copyLayer->addLine(this, 0, 1);
    setStartNode(this, 0, 0);
    setStartNode(copyLayer.get(), 1, 0);
}

void VkPoissonBlendLayer::onCommand() {
    // ping-pong 单数次
    paramet.iterationNum = paramet.iterationNum / 2 * 2 + 1;
    for (int32_t i = 0; i < paramet.iterationNum; i++) {
        int32_t pong = i % 2;
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        inTexs[1]->addBarrier(
            cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            pong == 0 ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT);
        outTexs[0]->addBarrier(
            cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            pong == 0 ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT);
        vkCmdPushConstants(cmd, layout->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t),
                           &pong);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computerPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout->pipelineLayout, 0, 1,
                                layout->descSets[0].data(), 0, 0);
        vkCmdDispatch(cmd, sizeX, sizeY, 1);
    }
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce