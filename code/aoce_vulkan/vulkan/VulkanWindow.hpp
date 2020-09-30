#pragma once

#include <memory>

#include "VulkanCommon.hpp"
#include "VulkanTexture.hpp"
#if WIN32
#include "../win32/Win32Window.hpp"
#elif __ANDROID__
#include <android/native_window_jni.h>
#endif
#include <functional>

#include "VulkanContext.hpp"

namespace aoce {
namespace vulkan {
#if WIN32
template class VKX_COMMON_EXPORT std::unique_ptr<VulkanTexture>;
template class VKX_COMMON_EXPORT std::vector<VkImage>;
template class VKX_COMMON_EXPORT std::vector<VkImageView>;
template class VKX_COMMON_EXPORT std::vector<VkFramebuffer>;
template class VKX_COMMON_EXPORT std::vector<VkCommandBuffer>;
template class VKX_COMMON_EXPORT std::unique_ptr<Win32Window>;
template class VKX_COMMON_EXPORT std::function<void(uint32_t)>;
template class VKX_COMMON_EXPORT std::function<void()>;
#endif

class VKX_COMMON_EXPORT VulkanWindow {
   private:
    class VulkanContext* context;
    VkSurfaceKHR surface;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::unique_ptr<VulkanTexture> depthTex;
    std::vector<VkImageView> views;
    VkCommandPool cmdPool;
    std::vector<VkFramebuffer> frameBuffers;
#if defined(_WIN32)
    std::unique_ptr<Win32Window> window;
#elif defined(__ANDROID__)
    android_app* androidApp;
#endif
    // 用于通知CPU,GPU上图像已经呈现出来
    VkFence presentFence;
    // 图像被获取,可以开始渲染
    VkSemaphore presentComplete;
    // 图像已经渲染,可以呈现
    VkSemaphore renderComplete;
    VkSubmitInfo submitInfo = {};
    VkPipelineStageFlags submitPipelineStages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    std::function<void(uint32_t)> onBuildCmd;
    std::function<void()> onPreDraw;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    bool bCanDraw = false;
    VkCommandBufferBeginInfo cmdBufferBeginInfo = {};
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    VkViewport viewport = {};
    VkRect2D scissor = {};
    VkClearValue clearValues[2];
    bool resizing = false;

   public:
    uint32_t graphicsQueueIndex = UINT32_MAX;
    uint32_t presentQueueIndex = UINT32_MAX;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_D16_UNORM;
    uint32_t width;
    uint32_t height;
    bool vsync = false;
    uint32_t imageCount;
    uint32_t currentIndex;
    VkRenderPass renderPass;
    std::vector<VkCommandBuffer> cmdBuffers;
    std::vector<VkImage> images;
    bool focused = false;
#if defined(__ANDROID__)
    std::function<void()> onInitVulkan;
#endif
   public:
    VulkanWindow(class VulkanContext* _context);
    ~VulkanWindow();
    // 没有外部窗口,自己创建
#if defined(_WIN32)
    void InitWindow(HINSTANCE inst, uint32_t _width, uint32_t height,
                    const char* appName);

    LRESULT handleMessage(UINT msg, WPARAM wparam, LPARAM lparam);
    // 根据窗口创建surface,并返回使用的queueIndex.
    void InitSurface(HINSTANCE inst, HWND windowHandle);
#elif defined(__ANDROID__)
    void InitWindow(android_app* app, std::function<void()> onInitVulkanAction);
    void InitSurface(ANativeWindow* window);
#endif
    void CreateSwipChain(VkDevice _device,
                         std::function<void(uint32_t)> onBuildCmdAction);

    void Run(std::function<void()> onPreDrawAction = nullptr);

   private:
    void reSwapChainBefore();
    void reSwapChainAfter();
    void createRenderPass();

    void tick();
};
}  // namespace vulkan
}  // namespace aoce