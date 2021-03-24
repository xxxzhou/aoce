#include "VkInputLayer.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkInputLayer::VkInputLayer(/* args */) {
    bInput = true;
    setUBOSize(12);
}

VkInputLayer::~VkInputLayer() {}

void VkInputLayer::onInitGraph() {
    std::string path = "glsl/inputv1.comp.spv";
    shader->loadShaderModule(context->device, path);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkInputLayer::onInitVkBuffer() {
    bUsePipe = true;
    sizeY = 1;
    int imageSize = inFormats[0].width * inFormats[0].height;
    // 如果是rgb-rgba,则先buffer转cs buffer,然后cs shader转rgba.
    // 不直接在cs shader用buf->tex,兼容性考虑cpu map/cs read权限.
    if (videoFormat.videoType == VideoType::rgb8) {
        // 每个线程组处理240个数据,一个线程拿buffer三个数据生成四个点
        sizeX = divUp(imageSize / 4, 240);
        inBufferX = std::make_unique<VulkanBuffer>();
        inBufferX->initResoure(BufferUsage::program, imageSize * 3,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else if (videoFormat.videoType == VideoType::argb8 ||
               videoFormat.videoType == VideoType::bgra8) {
        sizeX = divUp(imageSize, 240);
        inBufferX = std::make_unique<VulkanBuffer>();
        inBufferX->initResoure(BufferUsage::program, imageSize * 4,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else {
        bUsePipe = false;
    }
    int32_t size = inFormats[0].width * inFormats[0].height *
                   getImageTypeSize(inFormats[0].imageType);
    if (bUsePipe) {
        size = inBufferX->getBufferSize();
    }
    assert(size > 0);

    inBuffer = std::make_unique<VulkanBuffer>();
    inBuffer->initResoure(BufferUsage::store, size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, frameData);
    // 更新constBufCpu
    int imageIndex = -1;
    if (videoFormat.videoType == VideoType::rgb8) {
        imageIndex = 1;
    } else if (videoFormat.videoType == VideoType::bgra8) {
        imageIndex = 2;
    } else if (videoFormat.videoType == VideoType::argb8) {
        imageIndex = 3;
    }
    std::vector<int> ubo = {outFormats[0].width, outFormats[0].height,
                            imageIndex};
    updateUBO(ubo.data());
}

void VkInputLayer::onInitPipe() {
    if (bUsePipe) {
        outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        layout->updateSetLayout(0, 0, &inBufferX->descInfo,
                                &outTexs[0]->descInfo, &constBuf->descInfo);
        auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
            layout->pipelineLayout, shader->shaderStage);
        VK_CHECK_RESULT(vkCreateComputePipelines(
            context->device, context->pipelineCache, 1, &computePipelineInfo,
            nullptr, &computerPipeline));
    }
}

void VkInputLayer::onPreCmd() {
    // 不需要CS处理
    if (!bUsePipe) {
        outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT);
        context->bufferToImage(cmd, inBuffer.get(), outTexs[0].get());
    } else {
        VkBufferCopy copyRegion = {};
        copyRegion.size = inBufferX->descInfo.range;
        vkCmdCopyBuffer(cmd, inBuffer->buffer, inBufferX->buffer, 1,
                        &copyRegion);
        outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computerPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout->pipelineLayout, 0, 1,
                                layout->descSets[0].data(), 0, 0);
        vkCmdDispatch(cmd, sizeX, sizeY, 1);
    }
}

bool VkInputLayer::onFrame() {
    if (inBuffer) {
        inBuffer->upload(frameData);
        // inBuffer->submit();
        return true;
    }
    return false;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce