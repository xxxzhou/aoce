#pragma once
#include <Aoce.hpp>

#include "VulkanCommon.hpp"

#define VK_CHECK_RESULT(f)                                                \
    {                                                                     \
        VkResult res = (f);                                               \
        if (res != VK_SUCCESS) {                                          \
            logMessage(AOCE_LOG_ERROR, errorString(res).c_str()); \
            assert(res == VK_SUCCESS);                                    \
        }                                                                 \
    }

namespace aoce {
namespace vulkan {
AOCE_VULKAN_EXPORT const std::string getAssetPath();

// errorcode转显示
AOCE_VULKAN_EXPORT std::string errorString(VkResult errorCode);
// 物理显卡类型
AOCE_VULKAN_EXPORT std::string physicalDeviceTypeString(
    VkPhysicalDeviceType type);
// 创建一个vulkan实例
AOCE_VULKAN_EXPORT VkResult createInstance(VkInstance& instance,
                                          const char* appName);
// 得到所有物理显卡
AOCE_VULKAN_EXPORT VkResult enumerateDevice(
    VkInstance instance, std::vector<PhysicalDevice>& physicalDevices);
// 创建一个满足条件的逻辑设置,是否使用单独的计算通道
AOCE_VULKAN_EXPORT VkResult
createLogicalDevice(LogicalDevice& device, const PhysicalDevice& physicalDevice,
                    uint32_t queueFamilyIndex, bool bAloneCompute = false);
AOCE_VULKAN_EXPORT bool getMemoryTypeIndex(const PhysicalDevice& physicalDevice,
                                          uint32_t typeBits,
                                          VkFlags quirementsMaks,
                                          uint32_t& index);
AOCE_VULKAN_EXPORT int32_t getByteSize(VkFormat format);
// Load a SPIR-V shader (binary)
#if defined(__ANDROID__)
VkShaderModule loadShader(AAssetManager* assetManager, const char* fileName,
                          VkDevice device);
#else
AOCE_VULKAN_EXPORT VkShaderModule loadShader(const char* fileName,
                                            VkDevice device);
#endif

AOCE_VULKAN_EXPORT void changeLayout(
    VkCommandBuffer command, VkImage image, VkImageLayout oldLayout,
    VkImageLayout newLayout, VkPipelineStageFlags oldStageFlags,
    VkPipelineStageFlags newStageFlags,
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    VkAccessFlags newAccessFlags = 0);

}  // namespace vulkan
}  // namespace aoce
