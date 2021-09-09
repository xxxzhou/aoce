#pragma once
#include "vulkan/VulkanContext.hpp"

namespace aoce {
namespace vulkan {

// VulkanContext与VkPipeGraph构建了一个复杂的计算管线
// 但是有些时间只需要完成一些简单细碎的GPU操作

class VkCommand {
   private:
    /* data */
    VkDevice device = VK_NULL_HANDLE;
    // 管线缓存,加速管线创建
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkCommandBuffer computerCmd = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;

   public:
    VkCommand(/* args */);
    ~VkCommand();
};

}  // namespace vulkan
}  // namespace aoce

// vkCmdFillBuffer