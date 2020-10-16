#pragma once
#include <assert.h>
#include <vulkan/vulkan.h>

#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#elif defined(VK_USE_PLATFORM_WIN32_KHR)
#include <windows.h>
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
#include <android/asset_manager.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <android_native_app_glue.h>
#include <sys/system_properties.h>

#include "../android/vulkan_wrapper.h"
#endif

#ifdef _WIN32
#if defined(AOCE_VULKAN_EXPORT_DEFINE)
#define AOCE_VULKAN_EXPORT __declspec(dllexport)
#else
#define AOCE_VULKAN_EXPORT __declspec(dllimport)
#endif
#else
#define AOCE_VULKAN_EXPORT
#endif

namespace aoce {
namespace vulkan {
// 物理显卡封装
struct PhysicalDevice {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    // queueFlags包含每个通道VK_QUEUE_GRAPHICS_BIT/VK_QUEUE_COMPUTE_BIT能力表示
    std::vector<VkQueueFamilyProperties> queueFamilyProps;
    VkPhysicalDeviceMemoryProperties mempryProperties;
    VkPhysicalDeviceProperties properties;
    std::vector<uint32_t> queueGraphicsIndexs;
    std::vector<uint32_t> queueComputeIndexs;

    inline void clear() {
        queueFamilyProps.clear();
        queueGraphicsIndexs.clear();
        queueComputeIndexs.clear();
    }
};

// 逻辑显卡封装
struct LogicalDevice {
    VkDevice device = VK_NULL_HANDLE;
    // 选用的渲染通道索引
    uint32_t graphicsIndex;
    // 选用的计算通道过些
    uint32_t computeIndex;
};

struct FormatInfo {
    uint32_t size;
    uint32_t channelCount;
};
}  // namespace vulkan
}  // namespace aoce