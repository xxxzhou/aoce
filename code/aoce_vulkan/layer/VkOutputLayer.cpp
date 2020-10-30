#include "VkOutputLayer.hpp"

#include "../vulkan/VulkanManager.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkOutputLayer::VkOutputLayer(/* args */) { bOutput = true; }

VkOutputLayer::~VkOutputLayer() {
    if (outEvent) {
        vkDestroyEvent(context->device, outEvent, VK_NULL_HANDLE);
        outEvent = VK_NULL_HANDLE;
    }
}

void VkOutputLayer::onInitGraph() {
    VkEventCreateInfo eventInfo = {};
    eventInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventInfo.pNext = nullptr;
    eventInfo.flags = 0;
    vkCreateEvent(context->device, &eventInfo, nullptr, &outEvent);
    vkSetEvent(context->device, outEvent);
}

void VkOutputLayer::onInitVkBuffer() {
    int32_t size = outFormats[0].width * outFormats[0].height *
                   getImageTypeSize(outFormats[0].imageType);
    assert(size > 0);
    // CPU输出
    outBuffer = std::make_unique<VulkanBuffer>();
    outBuffer->initResoure(BufferUsage::store, size,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    cpuData.resize(size);
    // GPU输出
    const ImageFormat& format = outFormats[0];
    VkFormat vkft = ImageFormat2Vk(format.imageType);
    outTex = std::make_unique<VulkanTexture>();
    outTex->InitResource(outFormats[0].width, outFormats[0].height, vkft,
                         VK_IMAGE_USAGE_STORAGE_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         0);
}

bool VkOutputLayer::onFrame() {
    // outBuffer->upload(cpuData.data());
    onImageProcessHandle(outBuffer->getCpuData(), outFormats[0].width,
                         outFormats[0].height, 0);
    return true;
}

void VkOutputLayer::onPreCmd() {
    if (paramet.bCpu) {
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        context->imageToBuffer(cmd, inTexs[0].get(), outBuffer.get());
    }
    if (paramet.bGpu) {
        // vkCmdWaitEvents(cmd, 1, &outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT,
        //                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, 0,
        //                 nullptr, 0, nullptr);
        // vkCmdResetEvent(cmd, outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT);
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        outTex->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
        VulkanManager::copyImage(cmd, inTexs[0].get(), outTex.get());
        // vkCmdSetEvent(cmd, outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
}

void VkOutputLayer::outGpuTex(const VkOutGpuTex& outVkTex, int32_t outIndex) {
    if (!outTex || !outTex->image) {
        return;
    }
    // GPU输出
    VkCommandBuffer copyCmd = (VkCommandBuffer)outVkTex.commandbuffer;
    VkImage copyImage = (VkImage)outVkTex.image;
    auto res = vkGetEventStatus(context->device, outEvent);
    // assert(res != VK_EVENT_RESET);
    outTex->addBarrier(copyCmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
    VulkanManager::blitFillImage(copyCmd, outTex.get(), copyImage,
                                 outVkTex.width, outVkTex.height);
    // vkCmdSetEvent(cmd, outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce