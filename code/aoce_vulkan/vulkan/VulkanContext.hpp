#pragma once
#include "VulkanBuffer.hpp"
#include "VulkanCommon.hpp"
#include "VulkanHelper.hpp"
#include "VulkanTexture.hpp"

namespace aoce {
namespace vulkan {

// 把context与swapchain分开,用来Compture Shader做纯GPGPU运行时不需要swapchain
class AOCE_VULKAN_EXPORT VulkanContext {
   private:
    bool bAloneCompute = false;

   public:
    VkInstance instace;
    PhysicalDevice physicalDevice;
    LogicalDevice logicalDevice;
    VkQueue computeQueue;
    VkQueue graphicsQueue;
    // 管线缓存,加速管线创建
    VkPipelineCache pipelineCache;
    VkCommandBuffer computerCmd;
    VkCommandPool cmdPool;

   public:
    VulkanContext(/* args */);
    ~VulkanContext();

   public:
    void InitContext();

    bool CheckFormat(VkFormat format, VkFormatFeatureFlags feature, bool bLine);

    void BufferToImage(VkCommandBuffer cmd, const VulkanBuffer* buffer,
                       const VulkanTexture* texture);
    void ImageToBuffer(VkCommandBuffer cmd, const VulkanTexture* texture,
                       const VulkanBuffer* buffer);
    void BlitFillImage(VkCommandBuffer cmd, const VulkanTexture* src,
                       const VulkanTexture* dest);
    void BlitFillImage(
        VkCommandBuffer cmd, const VulkanTexture* src, VkImage dest,
        int32_t destWidth, int32_t destHeight,
        VkImageLayout destLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    bool IsAloneCompute() { return bAloneCompute; };
};
}  // namespace vulkan
}  // namespace aoce
