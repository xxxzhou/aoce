#pragma once
#include "VulkanCommon.hpp"

namespace aoce {
namespace vulkan {
// gpu buffer主要用于保存GPU数据,几种主要场景
enum class BufferUsage {
    other,
    // 1 用于和CPU交互,如UBO 可能需要一直更新
    store,
    // 2 用于GPU/GPU计算中的缓存数据,并不需要和CPU交互
    program,
    // 3 一次和CPU交互,后续一直用于GPU中,如纹理,模型VBO
    onestore,
};

// (CPU可写)VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
// (unmap后GPU最新)VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
// (CPU不可写,Computer shader)(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
// memory与view的关系应该是一对多,本类用于特殊的一对一情况
// 后期专门设计一个memory对应多个view的类。(可以一个mesh的IBO,VBO放一起,也可多Mesh放一块)
class AOCE_VULKAN_EXPORT VulkanBuffer {
   private:
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkBufferView view = VK_NULL_HANDLE;
    VkPipelineStageFlags stageFlags =
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;  // VK_PIPELINE_STAGE_HOST_BIT;
    VkAccessFlags accessFlags = VK_ACCESS_HOST_WRITE_BIT;
    int32_t bufferSize = 0;
    uint8_t* pData = nullptr;
    VkMemoryRequirements requires = {};

   public:
    VulkanBuffer();
    ~VulkanBuffer();

   public:
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDescriptorBufferInfo descInfo = {};

   public:
    void initResoure(BufferUsage usage,
                     uint32_t dataSize, VkBufferUsageFlags usageFlag,
                     uint8_t* cpuData = nullptr);
    void initView(VkFormat viewFormat);
    void release();

    void upload(const uint8_t* cpuData);
    // 复制pData到cpuData
    void download(uint8_t* cpuData);
    // 提交
    void submit();

    // 直接返回buffer关联pdata
    inline uint8_t* getCpuData() { return pData; };

    // Computer shader后,插入相关barrier,由读变写,由写变读,确保前面操作完成
    // 后续添加渲染管线与计算管线添加barrier的逻辑
    void addBarrier(VkCommandBuffer command, VkPipelineStageFlags newStageFlags,
                    VkAccessFlags newAccessFlags);
};
}  // namespace vulkan
}  // namespace aoce
