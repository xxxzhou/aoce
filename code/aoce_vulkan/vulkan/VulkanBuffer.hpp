#pragma once
#include "VulkanCommon.hpp"

namespace aoce {
namespace vulkan {
// gpu buffer主要用于保存GPU数据,几种主要场景
enum BufferUsage {
    VKS_BUFFER_USAGE_OTHER,
    // 1 用于和CPU交互,如UBO 可能需要一直更新
    VKS_BUFFER_USAGE_STORE,
    // 2 用于GPU/GPU计算中的缓存数据,并不需要和CPU交互
    VKS_BUFFER_USAGE_PROGRAM,
    // 3 一次和CPU交互,后续一直用于GPU中,如纹理,模型VBO
    VKS_BUFFER_USAGE_ONESTORE,
};

// (CPU可写)VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
// (unmap后GPU最新)VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
// (CPU不可写,Computer shader)(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
// memory与view的关系应该是一对多,本类用于特殊的一对一情况
// 后期专门设计一个memory对应多个view的类。(可以一个mesh的IBO,VBO放一起,也可多Mesh放一块)
class VKX_COMMON_EXPORT VulkanBuffer {
   private:
    VkDeviceMemory memory;
    VkDevice device;
    VkBufferView view;
    VkDescriptorBufferInfo descInfo;
    VkPipelineStageFlags stageFlags =
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;  // VK_PIPELINE_STAGE_HOST_BIT;
    VkAccessFlags accessFlags = VK_ACCESS_HOST_WRITE_BIT;

   public:
    VulkanBuffer();
    ~VulkanBuffer();

   public:
    VkBuffer buffer;

   public:
    void InitResource(class VulkanContext* context, uint32_t dataSize,
                      VkFormat viewFormat, VkBufferUsageFlags usageFlag,
                      VkMemoryPropertyFlags memoryFlag,
                      uint8_t* cpuData = nullptr);
    // Computer shader后,插入相关barrier,由读变写,由写变读,确保前面操作完成
    // 后续添加渲染管线与计算管线添加barrier的逻辑
    void AddBarrier(VkCommandBuffer command, VkPipelineStageFlags newStageFlags,
                    VkAccessFlags newAccessFlags);
};
}  // namespace common
}  // namespace vkx
