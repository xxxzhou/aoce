#include "VkYUV2RGBALayer.hpp"

#include "VkPipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkYUV2RGBALayer::VkYUV2RGBALayer(/* args */) { conBufSize = 12; }

VkYUV2RGBALayer::~VkYUV2RGBALayer() {}

void VkYUV2RGBALayer::onInitGraph() {
    int32_t yuvType = getYuvIndex(paramet.type);
    assert(yuvType > 0);
    std::string path = "./glsl/yuv2rgbaV1.comp.spv";
    if (yuvType > 3) {
        path = "./glsl/yuv2rgbaV2.comp.spv";
    }
    shader->loadShaderModule(context->device, path);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkYUV2RGBALayer::onUpdateParamet() {
    assert(getYuvIndex(paramet.type) > 0);
    pipeGraph->reset();
}

void VkYUV2RGBALayer::onInitLayer() {
    assert(getYuvIndex(paramet.type) > 0);
    // 带P/SP的格式由r8转rgba8
    inFormats[0].imageType = ImageType::r8;
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
    } else if (paramet.type == VideoType::yuv2I ||
               paramet.type == VideoType::yvyuI ||
               paramet.type == VideoType::yvyuI) {
        inFormats[0].imageType = ImageType::rgba8;
        // 一个线程处理二个点,yuyv四点组合成一个元素,和rgba类似
        outFormats[0].width = inFormats[0].width * 2;
        sizeX = divUp(inFormats[0].width, groupX);
        sizeY = divUp(inFormats[0].height, groupY);
    }
    // 更新constBufCpu
    std::vector<int> ubo = {outFormats[0].width, outFormats[0].height,
                            getYuvIndex(paramet.type)};
    memcpy(constBufCpu.data(), ubo.data(), conBufSize);
}

void VkYUV2RGBALayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &outTexs[0]->descInfo,
                            &constBuf->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1,
        &computePipelineInfo, nullptr, &computerPipeline));
}

void VkYUV2RGBALayer::onPreCmd() {
    inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
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