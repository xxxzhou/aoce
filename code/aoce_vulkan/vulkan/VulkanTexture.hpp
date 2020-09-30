#pragma once

#include "VulkanCommon.hpp"

namespace aoce {
namespace vulkan {
class VKX_COMMON_EXPORT VulkanTexture {
   private:
    VkDeviceMemory memory;
    VkDevice device;

   public:
    uint32_t width;
    uint32_t height;
    VkFormat format;
    VkSampler sampler;
    // image可能创建多个mipmap,多层级图像,view针对具体
    VkImageView view;
    VkImage image;
    // 这二个可以get出去,不能直接修改
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags accessFlags = VK_ACCESS_HOST_WRITE_BIT;
    VkDescriptorImageInfo descInfo;

   public:
    VulkanTexture();
    ~VulkanTexture();

   public:
    void InitResource(class VulkanContext* context, uint32_t width,
                      uint32_t height, VkFormat format,
                      VkImageUsageFlags usageFlag,
                      VkMemoryPropertyFlags memoryFlag,
                      uint8_t* cpuData = nullptr, uint8_t cpuPitch = 0);

    void AddBarrier(VkCommandBuffer command, VkImageLayout newLayout,
                    VkPipelineStageFlags newStageFlags,
                    VkAccessFlags newAccessFlags = 0);
};
}  // namespace common
}  // namespace vkx