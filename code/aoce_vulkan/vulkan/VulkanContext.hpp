#pragma once
#include "VulkanBuffer.hpp"
#include "VulkanCommon.hpp"
#include "VulkanHelper.hpp"
#include "VulkanTexture.hpp"

namespace aoce {
namespace vulkan {

// 把context与swapchain分开,用来Compture Shader做纯GPGPU运行时不需要swapchain
class VKX_COMMON_EXPORT VulkanContext {
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
    void CreateInstance(const char* appName);
    void CreateDevice(uint32_t queueIndex, bool bAloneCompute = false);

    bool CheckFormat(VkFormat format, VkFormatFeatureFlags feature, bool bLine);

    void BufferToImage(VkCommandBuffer cmd, const VulkanBuffer* buffer,
                       const VulkanTexture* texture);
    void BlitFillImage(VkCommandBuffer cmd, const VulkanTexture* src,
                       const VulkanTexture* dest);
    void BlitFillImage(
        VkCommandBuffer cmd, const VulkanTexture* src, VkImage dest,
        int32_t destWidth, int32_t destHeight,
        VkImageLayout destLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    bool IsAloneCompute() { return bAloneCompute; };
};
}  // namespace common
}  // namespace vkx
