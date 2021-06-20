#pragma once
#include "VulkanBuffer.hpp"
#include "VulkanCommon.hpp"
#include "VulkanHelper.hpp"
#include "VulkanTexture.hpp"

namespace aoce {
namespace vulkan {

// 封装一份GPGPU运行VkCommandBuffer
// 把context与swapchain分开,用来Compture Shader做纯GPGPU运行时不需要swapchain
class AOCE_VULKAN_EXPORT VulkanContext {
   public:
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkInstance instace = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    // VkQueue graphicsQueue = VK_NULL_HANDLE;
    // 管线缓存,加速管线创建
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkCommandBuffer computerCmd = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkSampler linearSampler = VK_NULL_HANDLE;
    VkSampler nearestSampler = VK_NULL_HANDLE;

   public:
    VulkanContext(/* args */);
    ~VulkanContext();

   private:
    void createSampler(bool bLinear, VkSampler& sampler);

   public:
    void initContext();

    bool checkFormat(VkFormat format, VkFormatFeatureFlags feature, bool bLine);

    void bufferToImage(VkCommandBuffer cmd, const VulkanBuffer* buffer,
                       const VulkanTexture* texture);
    void imageToBuffer(VkCommandBuffer cmd, const VulkanTexture* texture,
                       const VulkanBuffer* buffer);
    void imageToBuffer(VkCommandBuffer cmd, VkImage texture, VkBuffer buffer,
                       int32_t width, int32_t height);
    void blitFillImage(VkCommandBuffer cmd, const VulkanTexture* src,
                       const VulkanTexture* dest);
    void blitFillImage(
        VkCommandBuffer cmd, const VulkanTexture* src, VkImage dest,
        int32_t destWidth, int32_t destHeight,
        VkImageLayout destLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
};
}  // namespace vulkan
}  // namespace aoce
