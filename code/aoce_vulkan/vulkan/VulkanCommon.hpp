#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "aoce/Aoce.hpp"

#if defined(VK_USE_PLATFORM_WIN32_KHR)
#endif

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#include <android/asset_manager.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <android_native_app_glue.h>
#include <sys/system_properties.h>
// vulkan_wrapper需在vulkan之前,否则许多函数指针会重定义
#include "../android/vulkan_wrapper.h"
#endif

#include <vulkan/vulkan.h>

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
    // 一般来说,主机会有1-3个左右queueFamily,手机一般只一个.
    // 而第0个一般要满足graph/compute/present
    // queueFlags包含每个通道VK_QUEUE_GRAPHICS_BIT/VK_QUEUE_COMPUTE_BIT能力表示
    std::vector<VkQueueFamilyProperties> queueFamilyProps;
    // 物理显卡的内存显示
    VkPhysicalDeviceMemoryProperties mempryProperties;
    // 物理显示的属性
    VkPhysicalDeviceProperties properties;
    std::vector<uint32_t> queueGraphicsIndexs;
    std::vector<uint32_t> queueComputeIndexs;

    inline void clear() {
        queueFamilyProps.clear();
        queueGraphicsIndexs.clear();
        queueComputeIndexs.clear();
    }
};

// // 逻辑显卡封装
// struct LogicalDevice {
//     VkDevice device = VK_NULL_HANDLE;
//     // 选用的渲染通道索引
//     uint32_t graphicsIndex;
//     // 选用的计算通道过些
//     uint32_t computeIndex;
// };

struct FormatInfo {
    uint32_t size;
    uint32_t channelCount;
};
}  // namespace vulkan
}  // namespace aoce