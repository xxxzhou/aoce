#pragma once

#include <memory>
#include <mutex>

#include "VulkanCommon.hpp"
#include "VulkanTexture.hpp"

#if WIN32
#include "../win32/Win32Window.hpp"
#elif __ANDROID__
#include <android/native_activity.h>
#endif

#include <functional>
#include <mutex>

namespace aoce {
namespace vulkan {
#if WIN32
template class AOCE_VULKAN_EXPORT std::unique_ptr<VulkanTexture>;
template class AOCE_VULKAN_EXPORT std::vector<VkImage>;
template class AOCE_VULKAN_EXPORT std::vector<VkImageView>;
template class AOCE_VULKAN_EXPORT std::vector<VkFramebuffer>;
template class AOCE_VULKAN_EXPORT std::vector<VkCommandBuffer>;
template class AOCE_VULKAN_EXPORT std::unique_ptr<Win32Window>;
template class AOCE_VULKAN_EXPORT std::function<void(uint32_t)>;
template class AOCE_VULKAN_EXPORT std::function<void()>;
#endif

typedef std::function<void(uint32_t)> cmdExecuteHandle;

class AOCE_VULKAN_EXPORT VulkanWindow {
   private:
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::unique_ptr<VulkanTexture> depthTex;
    std::vector<VkImageView> views;
    VkCommandPool cmdPool;
    std::vector<VkFramebuffer> frameBuffers;
#if WIN32
    std::unique_ptr<Win32Window> window;
#elif __ANDROID__
    // std::mutex winMtx;
    // std::condition_variable winSignal;
    std::function<void()> onInitWindow = nullptr;
#endif
    // 用于动态添加执行列表
    std::vector<VkFence> addCmdFences;
    // 图像被获取,可以开始渲染
    VkSemaphore presentComplete;
    // 图像已经渲染,可以呈现
    VkSemaphore renderComplete;
    VkSubmitInfo submitInfo = {};
    VkPipelineStageFlags submitPipelineStages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // VkQueue graphicsQueue;
    VkQueue presentQueue;
    bool bCanDraw = false;
    VkCommandBufferBeginInfo cmdBufferBeginInfo = {};
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    VkViewport viewport = {};
    VkRect2D scissor = {};
    VkClearValue clearValues[2];
    // bool resizing = false;
    cmdExecuteHandle onCmdExecuteEvent;
    // 是否是自身创建的窗口,否则是用已有的窗口直接初始化surface
    bool bCreateWindow = false;
    bool bCreateSurface = false;
    bool focused = false;
    bool bPreCmd = false;
    bool vsync = false;
    std::mutex sizeMtx;

   public:
    // int32_t graphicsQueueIndex = UINT32_MAX;
    int32_t presentQueueIndex = UINT32_MAX;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_D16_UNORM;
    uint32_t width;
    uint32_t height;

    uint32_t imageCount;
    uint32_t currentIndex;
    VkRenderPass renderPass;
    std::vector<VkCommandBuffer> cmdBuffers;
    std::vector<VkImage> images;

   public:
    // preFram运行前确定,否则需要在运行后确定
    VulkanWindow(cmdExecuteHandle preCmd, bool preFram = false);

    ~VulkanWindow();
    // 没有外部窗口,自己创建
#if _WIN32
    void initWindow(HINSTANCE inst, uint32_t _width, uint32_t height,
                    const char* appName);

    LRESULT handleMessage(UINT msg, WPARAM wparam, LPARAM lparam);
    // 根据窗口创建surface,并返回使用的queueIndex.
    void initSurface(HINSTANCE inst, HWND windowHandle);
#elif defined(__ANDROID__)

    friend void handleAppCommand(android_app *app, int32_t cmd);
    // 一些参数限定只能在窗口主线程中使用
    void initWindow(std::function<void()> onInitWindow = nullptr);

    void initSurface(ANativeWindow *window);

#endif

    // 没有调用initWindow,无效
    void run();

    // 没有调用initWindow,直接用已有窗口initSurface,请在窗口的frame事件时调用
    void tick();

    void onSizeChange();

    bool windowCreate() { return bCreateSurface; };

   private:
    void createSwipChain(bool bInit = false);

    void reSwapChainBefore();

    void reSwapChainAfter();

    void createRenderPass();
};
}  // namespace vulkan
}  // namespace aoce