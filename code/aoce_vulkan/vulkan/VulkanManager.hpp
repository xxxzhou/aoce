#pragma once

#include "VulkanCommon.hpp"
#include "VulkanHelper.hpp"
#include "VulkanTexture.hpp"

namespace aoce {
namespace vulkan {

// 封装physical/logical device
class AOCE_VULKAN_EXPORT VulkanManager {
   public:
    static VulkanManager& Get();

   private:
    VulkanManager(/* args */);
    static VulkanManager* instance;

   public:
    ~VulkanManager();

   private:
    void releaseDevice();
    void releaseInstance();
    void onDeviceComplete();

   public:
    bool createInstance(const char* appName);
    // 查找一个不同于渲染通道的计算通道
    bool findAloneCompute(int32_t& familyIndex);
    bool createDevice(bool bAloneCompute = false);
    // presentIndex默认会选择graphicsIndex,如果graphicsIndex支持呈现,返回true
    bool findSurfaceQueue(VkSurfaceKHR surface, int32_t& presentIndex);
    // 第三方库有自己的vulkan上下文,并且aoce需要与之交互显存信息等,用第三方库替换
    void setVulkanContext(VkPhysicalDevice pdevice, VkDevice vdevice,
                          VkInstance vinstance = VK_NULL_HANDLE);

   public:
    static void blitFillImage(
        VkCommandBuffer cmd, const VulkanTexture* src, VkImage dest,
        int32_t destWidth, int32_t destHeight,
        VkImageLayout destLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    static void copyImage(VkCommandBuffer cmd, const VulkanTexture* src,
                          VkImage dest);
    static void copyImage(VkCommandBuffer cmd, VkImage src, VkImage dest,
                          int32_t width, int32_t height);
    // 选用的渲染通道索引
    int32_t graphicsIndex = -1;
    // 选用的计算通道过些
    int32_t computeIndex = -1;
    bool bAloneCompute = false;
    bool bInterpDx11 = false;
    bool bInterpGLES = false;
    bool bDebugMsg = false;
    bool bExternalContext = false;
    bool bExternalInstance = false;

   private:
    PhysicalDevice physical = {};

   public:
    VkInstance instace = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
#if __ANDROID__
    bool bAndroidHardware = false;
#endif
#if AOCE_DEBUG_TYPE
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
#endif
};

}  // namespace vulkan
}  // namespace aoce