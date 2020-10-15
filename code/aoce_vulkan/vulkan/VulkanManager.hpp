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
    VkInstance instace;
    PhysicalDevice physicalDevice;
    LogicalDevice logicalDevice;
    VkQueue computeQueue;
    VkQueue graphicsQueue;

#if defined(__ANDROID__)
    android_app* androidApp;
#endif
};

}  // namespace vulkan
}  // namespace aoce