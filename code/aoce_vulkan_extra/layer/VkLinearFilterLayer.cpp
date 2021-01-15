#include "VkLinearFilterLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLinearFilterLayer::VkLinearFilterLayer(/* args */) { conBufSize = 16; }

VkLinearFilterLayer::~VkLinearFilterLayer() {}

void VkLinearFilterLayer::onUpdateParamet() { pipeGraph->reset(); }

void VkLinearFilterLayer::onInitGraph() {
    std::string path = "glsl/filter2D.comp.spv";
    shader->loadShaderModule(context->device, path);

    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkLinearFilterLayer::onInitVkBuffer() {
    std::vector<int32_t> ubo = {paramet.kernelSizeX, paramet.kernelSizeY,
                                paramet.kernelSizeX / 2,
                                paramet.kernelSizeY / 2};
    memcpy(constBufCpu.data(), ubo.data(), conBufSize);

    int kernelSize = paramet.kernelSizeX * paramet.kernelSizeY;
    float kvulve = 1.0f / (float)kernelSize;
    std::vector<float> karray(kernelSize, kvulve);

    kernelBuffer = std::make_unique<VulkanBuffer>();
    kernelBuffer->initResoure(
        BufferUsage::onestore,
        paramet.kernelSizeX * paramet.kernelSizeY * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, (uint8_t*)karray.data());  
}

void VkLinearFilterLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &outTexs[0]->descInfo,
                            &constBuf->descInfo, &kernelBuffer->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce