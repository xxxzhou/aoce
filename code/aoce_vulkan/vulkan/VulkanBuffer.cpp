#include "VulkanBuffer.hpp"

#include "VulkanContext.hpp"
#include "VulkanHelper.hpp"
#include "VulkanManager.hpp"
#if __ANDROID__
#include "../android/vulkan_wrapper.h"
#endif

namespace aoce {
namespace vulkan {
VulkanBuffer::VulkanBuffer() {}

VulkanBuffer::~VulkanBuffer() { release(); }

void VulkanBuffer::initResoure(BufferUsage usage,
                               uint32_t dataSize, VkBufferUsageFlags usageFlag,
                               uint8_t* cpuData) {
    // 先释放可能已经在的资源
    this->release();
    // 重新生成
    this->device = VulkanManager::Get().device;
    this->bufferSize = dataSize;
    // 生成buffer
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.pNext = nullptr;
    bufInfo.usage = usageFlag;  //用来指定用于VBO/IBO/UBO等
    bufInfo.size = dataSize;
    bufInfo.queueFamilyIndexCount = 0;
    bufInfo.pQueueFamilyIndices = nullptr;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufInfo.flags = 0;
    // 生成一个buffer标识(可以多个标识到同一个显存空间DeviceMemory里,标识用来指定VBO/IBO/UBO等)
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufInfo, nullptr, &buffer));
    // Buffer存入位置
    VkMemoryPropertyFlags memoryFlag = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (usage == BufferUsage::onestore || usage == BufferUsage::store) {
        memoryFlag = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    vkGetBufferMemoryRequirements(device, buffer, &requires);
    uint32_t memoryTypeIndex = 0;
    bool getIndex = getMemoryTypeIndex(requires.memoryTypeBits, memoryFlag,
                                       memoryTypeIndex);
    assert(getIndex == true);
    VkMemoryAllocateInfo memoryInfo = {};
    memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryInfo.pNext = nullptr;
    memoryInfo.memoryTypeIndex = memoryTypeIndex;
    memoryInfo.allocationSize = requires.size;
    // 生成设备显存空间
    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryInfo, nullptr, &memory));
    if (cpuData) {
        VK_CHECK_RESULT(
            vkMapMemory(device, memory, 0, dataSize, 0, (void**)&pData));
        memcpy(pData, cpuData, dataSize);
        // 这段逻辑需要再考虑下
        if ((memoryFlag & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange memRange;
            memRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            memRange.pNext = nullptr;
            memRange.memory = memory;
            memRange.offset = 0;
            memRange.size = requires.size;
            VK_CHECK_RESULT(vkFlushMappedMemoryRanges(device, 1, &memRange));
        }
        vkUnmapMemory(device, memory);
        pData = nullptr;
    }
    VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));

    descInfo.buffer = buffer;
    descInfo.offset = 0;
    descInfo.range = dataSize;

    if (usage == BufferUsage::store) {
        VK_CHECK_RESULT(
            vkMapMemory(device, memory, 0, dataSize, 0, (void**)&pData));
    }
}

void VulkanBuffer::initView(VkFormat viewFormat) {
    VkBufferViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.buffer = buffer;
    viewInfo.format = viewFormat;
    viewInfo.offset = 0;
    viewInfo.range = requires.size;
    VK_CHECK_RESULT(vkCreateBufferView(device, &viewInfo, nullptr, &view));
}

void VulkanBuffer::upload(const uint8_t* cpuData) {
    if (pData == nullptr) {
        VK_CHECK_RESULT(
            vkMapMemory(device, memory, 0, bufferSize, 0, (void**)&pData));
    }
    memcpy(pData, cpuData, bufferSize);
}

void VulkanBuffer::download(uint8_t* cpuData) {
    if (pData == nullptr) {
        VK_CHECK_RESULT(
            vkMapMemory(device, memory, 0, bufferSize, 0, (void**)&pData));
    }
    memcpy(cpuData, pData, bufferSize);
}

void VulkanBuffer::submit() {
    if (pData) {
        vkUnmapMemory(device, memory);
        pData = nullptr;
    }
}

void VulkanBuffer::release() {
    if (pData != nullptr) {
        vkUnmapMemory(device, memory);
        pData = nullptr;
    }
    if (view) {
        vkDestroyBufferView(device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (buffer) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
    if (memory) {
        vkFreeMemory(device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
    }
}

void VulkanBuffer::addBarrier(VkCommandBuffer command,
                              VkPipelineStageFlags newStageFlags,
                              VkAccessFlags newAccessFlags) {
    VkPipelineStageFlags oldStageFlags = stageFlags;
    VkAccessFlags oldAccessFlags = accessFlags;

    VkBufferMemoryBarrier bufBarrier = {};
    bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;

    bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.buffer = buffer;
    bufBarrier.offset = 0;
    bufBarrier.size = VK_WHOLE_SIZE;
    bufBarrier.srcAccessMask = oldAccessFlags;
    bufBarrier.dstAccessMask = newAccessFlags;
    vkCmdPipelineBarrier(command, oldStageFlags, newStageFlags, 0, 0, nullptr,
                         1, &bufBarrier, 0, nullptr);
    stageFlags = newStageFlags;
    accessFlags = newAccessFlags;
}

}  // namespace vulkan
}  // namespace aoce