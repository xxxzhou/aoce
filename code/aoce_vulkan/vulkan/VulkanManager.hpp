#pragma once

#include "VulkanCommon.hpp"
#include "VulkanHelper.hpp"

namespace aoce {
namespace vulkan {

class AOCE_VULKAN_EXPORT VulkanManager {
   public:
    static VulkanManager& Get();

   private:
    VulkanManager(/* args */);
    static VulkanManager* instance;

   public:
    ~VulkanManager();

   public:
    void CreateInstance(const char* appName);
    void CreateDevice(uint32_t queueIndex, bool bAloneCompute = false);

   public:
    VkInstance instace = VK_NULL_HANDLE;
    PhysicalDevice physicalDevice = {};
    // 一般来说,主机会有1-3个左右queueFamily,手机一般只一个.
    // 而第0个一般要满足graph/compute/present
    LogicalDevice logicalDevice = {};
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;

#if defined(__ANDROID__)
    android_app* androidApp;
#endif
};

}  // namespace vulkan
}  // namespace aoce