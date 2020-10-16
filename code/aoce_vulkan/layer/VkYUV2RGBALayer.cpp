#include "VkYUV2RGBALayer.hpp"

#include "VkPipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkYUV2RGBALayer::VkYUV2RGBALayer(/* args */) {}

VkYUV2RGBALayer::~VkYUV2RGBALayer() {}

void VkYUV2RGBALayer::onInitGraph() {
    std::string path = "./glsl/yuv2rgbaV1.comp.spv";
    shader->loadShaderModule(context->logicalDevice.device, path);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->AddSetLayout(items);
    layout->GenerateLayout();
}

void VkYUV2RGBALayer::onUpdateParamet() {
    assert(bYuv(paramet.type));
    pipeGraph->reset();
}

void VkYUV2RGBALayer::onInitLayer() {
    assert(bYuv(paramet.type));
    // 带P/SP的格式由r8转rgba8
    outFormats[0].imageType = ImageType::rgba8;
    if (paramet.type == VideoType::nv12 || paramet.type == VideoType::yuv420P) {
        outFormats[0].height = inFormats[0].height * 2 / 3;
        // 一个线程处理四个点
        sizeX = divUp(outFormats[0].width, 2 * groupX);
        sizeY = divUp(outFormats[0].height, 2 * groupY);
    } else if (paramet.type == VideoType::yuy2P) {
        outFormats[0].height = inFormats[0].height / 2;
        // 一个线程处理二个点
        sizeX = divUp(outFormats[0].width, 2 * groupX);
        sizeY = divUp(outFormats[0].height, groupY);
    }
}

void VkYUV2RGBALayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->UpdateSetLayout(0, 0, &inTexs[0]->descInfo, &outTexs[0]->descInfo);
    auto computePipelineInfo = VulkanPipeline::CreateComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->logicalDevice.device, context->pipelineCache, 1,
        &computePipelineInfo, nullptr, &computerPipeline));
}

void VkYUV2RGBALayer::onPreCmd() {
    inTexs[0]->AddBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    outTexs[0]->AddBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce