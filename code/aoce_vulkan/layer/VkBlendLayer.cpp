#include "VkBlendLayer.hpp"

#include <math.h>

#include "VkPipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkBlendLayer::VkBlendLayer(/* args */) {
    setUBOSize(sizeof(VkBlendParamet));
    inCount = 2;
    outCount = 1;    
}

VkBlendLayer::~VkBlendLayer() {}

void VkBlendLayer::onUpdateParamet() {
    parametTransform();
    // 运行时更新UBO
    bParametChange = true;
}

void VkBlendLayer::onInitGraph() {
    std::string path = "glsl/resize.comp.spv";
    shader->loadShaderModule(context->device, path);
    // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkBlendLayer::parametTransform() {
    vkParamet.centerX = paramet.centerX;
    vkParamet.centerY = paramet.centerY;
    vkParamet.width = paramet.width;
    vkParamet.height = paramet.height;
    vkParamet.opacity = std::max(0.f, std::min(1.0f, paramet.alaph));
    if (inFormats.size() > 1) {
        vkParamet.fx = inFormats[1].width / vkParamet.width;
        vkParamet.fy = inFormats[1].height / vkParamet.height;
    }
    memcpy(constBufCpu.data(), &vkParamet, conBufSize);
}

void VkBlendLayer::onInitLayer() {
    VkLayer::onInitLayer();

    parametTransform();
}

void VkBlendLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    inTexs[1]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    inTexs[1]->descInfo.sampler = context->linearSampler;
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &inTexs[1]->descInfo,
                            &outTexs[0]->descInfo, &constBuf->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
