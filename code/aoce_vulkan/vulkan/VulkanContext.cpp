#include "VulkanContext.hpp"

#include "VulkanManager.hpp"
namespace aoce {
namespace vulkan {
VulkanContext::VulkanContext(/* args */) {}

VulkanContext::~VulkanContext() {
    if (pipelineCache) {
        vkDestroyPipelineCache(logicalDevice.device, pipelineCache, nullptr);
    }
}

void VulkanContext::InitContext() {
    // 得到全局设置
    this->instace = VulkanManager::Get().instace;
    this->physicalDevice = VulkanManager::Get().physicalDevice;
    this->logicalDevice = VulkanManager::Get().logicalDevice;
    this->computeQueue = VulkanManager::Get().computeQueue;
    this->graphicsQueue = VulkanManager::Get().graphicsQueue;
    this->bAloneCompute =
        logicalDevice.computeIndex != logicalDevice.graphicsIndex;
    // 得到当前graph需要的VkCommandBuffer
    VkPipelineCacheCreateInfo pipelineCacheInfo = {};
    pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK_RESULT(vkCreatePipelineCache(
        logicalDevice.device, &pipelineCacheInfo, nullptr, &pipelineCache));
    // context和呈现渲染相应command分开
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = logicalDevice.computeIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(logicalDevice.device, &cmdPoolInfo,
                                        nullptr, &cmdPool));
    VkCommandBufferAllocateInfo cmdBufInfo = {};
    cmdBufInfo.commandPool = cmdPool;
    cmdBufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(logicalDevice.device, &cmdBufInfo,
                                             &computerCmd));
}

bool VulkanContext::CheckFormat(VkFormat format, VkFormatFeatureFlags feature,
                                bool bLine) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice.physicalDevice, format,
                                        &props);
    if (bLine) {
        return (props.linearTilingFeatures & feature) == feature;
    } else {
        return (props.optimalTilingFeatures & feature) == feature;
    }
}

void VulkanContext::BufferToImage(VkCommandBuffer cmd,
                                  const VulkanBuffer* buffer,
                                  const VulkanTexture* texture) {
    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    // copyRegion.bufferRowLength = texture->width *
    // getByteSize(texture->format); copyRegion.bufferImageHeight =
    // texture->height;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent.width = texture->width;
    copyRegion.imageExtent.height = texture->height;
    copyRegion.imageExtent.depth = 1;

    // buffer to image
    vkCmdCopyBufferToImage(cmd, buffer->buffer, texture->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);
}

void VulkanContext::ImageToBuffer(VkCommandBuffer cmd,
                                  const VulkanTexture* texture,
                                  const VulkanBuffer* buffer) {
    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    // copyRegion.bufferRowLength = texture->width *
    // getByteSize(texture->format); copyRegion.bufferImageHeight =
    // texture->height;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent.width = texture->width;
    copyRegion.imageExtent.height = texture->height;
    copyRegion.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cmd, texture->image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->buffer,
                           1, &copyRegion);
}

void VulkanContext::BlitFillImage(VkCommandBuffer cmd, const VulkanTexture* src,
                                  const VulkanTexture* dest) {
    BlitFillImage(cmd, src, dest->image, dest->width, dest->height,
                  dest->layout);
}

void VulkanContext::BlitFillImage(VkCommandBuffer cmd, const VulkanTexture* src,
                                  VkImage dest, int32_t destWidth,
                                  int32_t destHeight,
                                  VkImageLayout destLayout) {
    VkImageBlit region;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.mipLevel = 0;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount = 1;
    region.srcOffsets[0].x = 0;
    region.srcOffsets[0].y = 0;
    region.srcOffsets[0].z = 0;
    region.srcOffsets[1].x = src->width;
    region.srcOffsets[1].y = src->height;
    region.srcOffsets[1].z = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.mipLevel = 0;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount = 1;
    region.dstOffsets[0].x = 0;
    region.dstOffsets[0].y = 0;
    region.dstOffsets[0].z = 0;
    region.dstOffsets[1].x = destWidth;
    region.dstOffsets[1].y = destHeight;
    region.dstOffsets[1].z = 1;
    // 要将源图像的区域复制到目标图像中，并可能执行格式转换，任意缩放和过滤
    vkCmdBlitImage(cmd, src->image, src->layout, dest, destLayout, 1, &region,
                   VK_FILTER_LINEAR);
}

}  // namespace vulkan
}  // namespace aoce