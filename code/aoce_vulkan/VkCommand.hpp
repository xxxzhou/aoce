#pragma once
#include "vulkan/VulkanContext.hpp"

namespace aoce {
namespace vulkan {

// VulkanContext与VkPipeGraph构建了一个复杂的计算管线
// 但是有些时间只需要完成一些简单细碎的GPU操作
// 注意,所有API调用请保持在一个线程中
class AOCE_VULKAN_EXPORT VkCommand {
   private:
    /* data */
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;

    // submit后,除非调用过reset,否则后面record不会起作用
    bool bRecord = false;
    bool bSubmit = false;

   public:
    VkCommand(/* args */);
    VkCommand(VkDevice vdevice);
    ~VkCommand();

   private:
    void beginRecord();
    void endRecord();

   public:
    void barrier(VkBuffer buffer, VkPipelineStageFlags stageFlag,
                 VkAccessFlags assessFlag, VkPipelineStageFlags oldStageFlag,
                 VkAccessFlags oldAssessFlag);
    void record(VkBuffer src, VkBuffer dest, int32_t destOffset, int32_t size);
    void submit();
    void reset();
};

}  // namespace vulkan
}  // namespace aoce

// vkCmdFillBuffer