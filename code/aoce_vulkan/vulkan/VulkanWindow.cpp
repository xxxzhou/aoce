#include "VulkanWindow.hpp"

#include <AoceManager.hpp>
#include <array>

#include "VulkanCommon.hpp"
#include "VulkanContext.hpp"
#include "VulkanHelper.hpp"
#include "VulkanManager.hpp"
#if __ANDROID__
#include "../android/vulkan_wrapper.h"
#endif

namespace aoce {
namespace vulkan {
VulkanWindow::VulkanWindow(cmdExecuteHandle preCmd, bool preFram) {
    this->instance = VulkanManager::Get().instace;
    this->physicalDevice = VulkanManager::Get().physicalDevice;
    this->device = VulkanManager::Get().device;
    this->onCmdExecuteEvent = preCmd;
    this->bPreCmd = preFram;
}

VulkanWindow::~VulkanWindow() {
    if (swapChain) {
        for (uint32_t i = 0; i < imageCount; i++) {
            vkDestroyImageView(device, views[i], nullptr);
            vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
            vkDestroyFence(device, addCmdFences[i], nullptr);
        }
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        vkDestroySemaphore(device, presentComplete, nullptr);
        vkDestroySemaphore(device, renderComplete, nullptr);
        vkFreeCommandBuffers(device, cmdPool,
                             static_cast<uint32_t>(cmdBuffers.size()),
                             cmdBuffers.data());
        vkDestroyCommandPool(device, cmdPool, nullptr);
        swapChain = VK_NULL_HANDLE;
    }
    if (surface) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
}

#if WIN32
void VulkanWindow::initWindow(HINSTANCE inst, uint32_t _width, uint32_t _height,
                              const char* appName) {
    this->width = _width;
    this->height = _height;
    // 创建窗口
    window = std::make_unique<Win32Window>();
    HWND hwnd = window->initWindow(inst, width, height, appName, this);
    // 得到合适的queueIndex
    this->initSurface(inst, hwnd);
    bCreateWindow = true;
}

LRESULT VulkanWindow::handleMessage(UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
        case WM_SIZE: {
            this->width = LOWORD(lparam);
            this->height = HIWORD(lparam);
            if (width > 0 && height > 0) {
                onSizeChange();
            }
        } break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(window->hwnd, msg, wparam, lparam);
            break;
    }
    return 0;
}
#elif __ANDROID__

void handleAppCommand(android_app* app, int32_t cmd) {
    assert(app->userData != NULL);
    VulkanWindow* window = reinterpret_cast<VulkanWindow*>(app->userData);
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            LOGI("\n");
            LOGI("=================================================");
            LOGI("          The sample ran successfully!!");
            LOGI("=================================================");
            LOGI("\n");
            window->initSurface(app->window);
            // window->winSignal.notify_all();
            if (window->onInitWindow != nullptr) {
                window->onInitWindow();
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            break;
        case APP_CMD_WINDOW_RESIZED:
            window->onSizeChange();
            break;
        case APP_CMD_LOST_FOCUS:
            LOGI("APP_CMD_LOST_FOCUS");
            window->focused = false;
            break;
        case APP_CMD_GAINED_FOCUS:
            LOGI("APP_CMD_GAINED_FOCUS");
            window->focused = true;
            break;
        case APP_CMD_STOP:
            ANativeActivity_finish(app->activity);
            break;
        default:
            LOGI("event not handled: %d", cmd);
    }
}

void VulkanWindow::initWindow(std::function<void()> onInitWindow) {
    this->onInitWindow = onInitWindow;
    AoceManager::Get().getApp()->userData = this;
    AoceManager::Get().getApp()->onAppCmd = handleAppCommand;
    bCreateWindow = true;
}

#endif

#if defined(_WIN32)
void VulkanWindow::initSurface(HINSTANCE inst, HWND windowHandle)
#elif defined(__ANDROID__)
void VulkanWindow::initSurface(ANativeWindow* window)
#endif
{
    VkResult ret = VK_SUCCESS;
#if defined(_WIN32)
    VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceCreateInfo.hinstance = inst;
    surfaceCreateInfo.hwnd = (HWND)windowHandle;
    VK_CHECK_RESULT(vkCreateWin32SurfaceKHR(instance, &surfaceCreateInfo,
                                            nullptr, &surface));
#elif defined(__ANDROID__)
    VkAndroidSurfaceCreateInfoKHR surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    surfaceCreateInfo.window = window;
    surfaceCreateInfo.flags = 0;
    surfaceCreateInfo.pNext = nullptr;
    ret =
        vkCreateAndroidSurfaceKHR(instance, &surfaceCreateInfo, NULL, &surface);
#endif
    bool bfind =
        VulkanManager::Get().findSurfaceQueue(surface, presentQueueIndex);
    assert(presentQueueIndex >= 0);
    if (!bfind) {
        logMessage(LogLevel::warn, "presentIndex not equal graphicsIndex");
    }
    //查找surf支持的显示格式
    uint32_t formatCount;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> surfFormats(formatCount);
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice, surface, &formatCount, surfFormats.data()));
    if (formatCount == 1 && surfFormats[0].format == VK_FORMAT_UNDEFINED) {
        format = surfFormats[0];
        format.format = VK_FORMAT_B8G8R8A8_UNORM;
    } else {
        assert(formatCount >= 1);
        bool bfind = false;
        for (auto& surf : surfFormats) {
            if (surf.format == VK_FORMAT_B8G8R8A8_UNORM ||
                surf.format == VK_FORMAT_R8G8B8A8_UNORM) {
                format = surf;
                bfind = true;
                break;
            }
        }
        if (!bfind) {
            format = surfFormats[0];
        }
    }
    // 创建semaphore 与 submitInfo,同步渲染与命令执行的顺序
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    // 确保显示图像后再提交渲染命令,vkAcquireNextImageKHR用来presentComplete使presentComplete上锁
    VK_CHECK_RESULT(
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &presentComplete));
    // 确保提交与执行命令后,才能执行显示图像
    VK_CHECK_RESULT(
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderComplete));
    // 用于渲染队列里确定等待与发送信号
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pWaitDstStageMask = &submitPipelineStages;
    submitInfo.waitSemaphoreCount = 1;
    // 用于指定队列开始执行前需要等待的信号量,以及需要等待的管线阶段
    submitInfo.pWaitSemaphores = &presentComplete;
    submitInfo.signalSemaphoreCount = 1;
    // 用于缓冲命令执行结束后发出信号的信号量对象
    submitInfo.pSignalSemaphores = &renderComplete;
    // 得到当前使用的queue
    vkGetDeviceQueue(device, presentQueueIndex, 0, &presentQueue);
    // reSwapChain主要是重新生成由widht/height改变的资源 cmdPool
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = presentQueueIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(
        vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
    // 创建窗口用的renderpass
    createRenderPass();
    createSwipChain(true);
    bCreateSurface = true;
}

void VulkanWindow::createSwipChain(bool bInit) {
    // std::lock_guard<std::mutex> mtx_locker(sizeMtx);
    bCanDraw = false;
    // 得到imagecount
    this->reSwapChainBefore();
    // 创建swapchain以及对应的image
    this->reSwapChainAfter();
    if (bInit) {
        // imageCount一般来说不会变动
        cmdBuffers.resize(imageCount);
        VkCommandBufferAllocateInfo cmdBufInfo = {};
        cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufInfo.commandPool = cmdPool;
        cmdBufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufInfo.commandBufferCount = (uint32_t)cmdBuffers.size();
        VK_CHECK_RESULT(
            vkAllocateCommandBuffers(device, &cmdBufInfo, cmdBuffers.data()));
        // 开始组合build
        if (this->bPreCmd) {
            for (uint32_t i = 0; i < imageCount; i++) {
                vkBeginCommandBuffer(cmdBuffers[i], &cmdBufferBeginInfo);
                vkCmdSetViewport(cmdBuffers[i], 0, 1, &viewport);
                vkCmdSetScissor(cmdBuffers[i], 0, 1, &scissor);
                renderPassBeginInfo.framebuffer = frameBuffers[i];
                // renderpass关联渲染目标
                // vkCmdBeginRenderPass(cmdBuffers[i], &renderPassBeginInfo,
                //                      VK_SUBPASS_CONTENTS_INLINE);
                if (onCmdExecuteEvent) {
                    onCmdExecuteEvent(i);
                }
                // vkCmdEndRenderPass(cmdBuffers[i]);
                vkEndCommandBuffer(cmdBuffers[i]);
            }
        }
        addCmdFences.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; i++) {
            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            // 默认是有信号
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(device, &fenceInfo, nullptr, &addCmdFences[i]);
        }
    }
    bCanDraw = true;
}

void VulkanWindow::reSwapChainBefore() {
    VkSwapchainKHR oldSwapchain = swapChain;
    // Get physical device surface properties and formats
    VkSurfaceCapabilitiesKHR surfCapabilities;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physicalDevice, surface, &surfCapabilities));
    // 得到交换链支持的所有Mode
    uint32_t presentModeCount;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, nullptr));
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, presentModes.data()));

    VkExtent2D swapchainExtent = {};
    // width and height are either both 0xFFFFFFFF, or both not 0xFFFFFFFF.
    if (surfCapabilities.currentExtent.width == 0xFFFFFFFF) {
        // If the surface size is undefined, the size is set to
        // the size of the images requested.
        swapchainExtent.width = width;
        swapchainExtent.height = height;
        if (swapchainExtent.width < surfCapabilities.minImageExtent.width) {
            swapchainExtent.width = surfCapabilities.minImageExtent.width;
        } else if (swapchainExtent.width >
                   surfCapabilities.maxImageExtent.width) {
            swapchainExtent.width = surfCapabilities.maxImageExtent.width;
        }

        if (swapchainExtent.height < surfCapabilities.minImageExtent.height) {
            swapchainExtent.height = surfCapabilities.minImageExtent.height;
        } else if (swapchainExtent.height >
                   surfCapabilities.maxImageExtent.height) {
            swapchainExtent.height = surfCapabilities.maxImageExtent.height;
        }
    } else {
        // If the surface size is defined, the swap chain size must match
        swapchainExtent = surfCapabilities.currentExtent;
    }
    // 得到交换链要求的长宽,surface大小变动后,要重新获得
    this->width = swapchainExtent.width;
    this->height = swapchainExtent.height;
    // present需要支持FIFO
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync) {
        for (size_t i = 0; i < presentModeCount; i++) {
            if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
                swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
            if ((swapchainPresentMode != VK_PRESENT_MODE_MAILBOX_KHR) &&
                (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)) {
                swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
            }
        }
    }
    uint32_t desiredNumberOfSwapchainImages =
        surfCapabilities.minImageCount + 1;
    if ((surfCapabilities.maxImageCount > 0) &&
        (desiredNumberOfSwapchainImages > surfCapabilities.maxImageCount)) {
        desiredNumberOfSwapchainImages = surfCapabilities.maxImageCount;
    }
    // Find the transformation of the surface
    VkSurfaceTransformFlagsKHR preTransform =
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    if (surfCapabilities.supportedTransforms &
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        // We prefer a non-rotated transform
        preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        preTransform = surfCapabilities.currentTransform;
    }
    // Find a supported composite alpha mode - one of these is guaranteed to be
    // set
    VkCompositeAlphaFlagBitsKHR compositeAlpha =
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    VkCompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (uint32_t i = 0;
         i < sizeof(compositeAlphaFlags) / sizeof(compositeAlphaFlags[0]);
         i++) {
        if (surfCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
            compositeAlpha = compositeAlphaFlags[i];
            break;
        }
    }
    // crate swapchain
    VkSwapchainCreateInfoKHR swapchainInfo = {};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.pNext = nullptr;
    swapchainInfo.surface = surface;
    swapchainInfo.minImageCount = desiredNumberOfSwapchainImages;
    swapchainInfo.imageFormat = format.format;
    swapchainInfo.imageColorSpace = format.colorSpace;
    swapchainInfo.imageExtent = {swapchainExtent.width, swapchainExtent.height};
    swapchainInfo.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
    swapchainInfo.compositeAlpha = compositeAlpha;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.presentMode = swapchainPresentMode;
    swapchainInfo.oldSwapchain = oldSwapchain;
    swapchainInfo.clipped = VK_FALSE;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainInfo.queueFamilyIndexCount = 0;
    swapchainInfo.pQueueFamilyIndices = nullptr;
    // 如果二个不同队列,我们需要CONCURRENT共享不同队列传输
    int graphicsQueueIndex = VulkanManager::Get().graphicsIndex;
    if (graphicsQueueIndex != presentQueueIndex) {
        uint32_t queueFamilyIndices[2] = {(uint32_t)graphicsQueueIndex,
                                          (uint32_t)presentQueueIndex};
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    VK_CHECK_RESULT(
        vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapChain));
    // 自动销毁老的资源,重新生成
    depthTex = std::make_unique<VulkanTexture>();
    depthTex->InitResource(width, height, depthFormat,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // 销毁老的资源
    if (oldSwapchain != VK_NULL_HANDLE) {
        for (uint32_t i = 0; i < imageCount; i++) {
            vkDestroyImageView(device, views[i], nullptr);
            vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
        }
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
        views.clear();
        frameBuffers.clear();
    }
    VK_CHECK_RESULT(
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr));
    images.resize(imageCount);
    VK_CHECK_RESULT(
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, images.data()));
    views.resize(imageCount);
    frameBuffers.resize(imageCount);
}

void VulkanWindow::reSwapChainAfter() {
    // 创建swap image view
    VkImageViewCreateInfo colorAttachmentView = {};
    colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    colorAttachmentView.pNext = nullptr;
    colorAttachmentView.format = format.format;
    colorAttachmentView.components = {
        VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
        VK_COMPONENT_SWIZZLE_A};
    colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorAttachmentView.subresourceRange.baseMipLevel = 0;
    colorAttachmentView.subresourceRange.levelCount = 1;
    colorAttachmentView.subresourceRange.baseArrayLayer = 0;
    colorAttachmentView.subresourceRange.layerCount = 1;
    colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorAttachmentView.flags = 0;

    VkImageView attachments[2];
    attachments[1] = depthTex->view;
    // 创建FBO
    VkFramebufferCreateInfo frameBufferCreateInfo = {};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.pNext = NULL;
    frameBufferCreateInfo.renderPass = renderPass;
    frameBufferCreateInfo.attachmentCount = 2;
    // 需要满足renderpass创建指定的VkAttachmentDescription
    frameBufferCreateInfo.pAttachments = attachments;
    frameBufferCreateInfo.width = width;
    frameBufferCreateInfo.height = height;
    frameBufferCreateInfo.layers = 1;
    for (uint32_t i = 0; i < imageCount; i++) {
        colorAttachmentView.image = images[i];
        VK_CHECK_RESULT(vkCreateImageView(device, &colorAttachmentView, nullptr,
                                          &views[i]));
        attachments[0] = views[i];
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo,
                                            nullptr, &frameBuffers[i]));
    }
    //因大小变化,重新组合相应buffer
    // 创建cmdBufferBeginInfo
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // 创建renderpassbeginInfo
    // VkClearValue clearValues[2];
    clearValues[0].color = {{0.0f, 1.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.extent = {width, height};
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;
    // viewport
    viewport.width = (float)width;
    viewport.height = (float)height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    // scissor
    scissor.extent = {width, height};
    scissor.offset = {0, 0};
}

void VulkanWindow::createRenderPass() {
    std::array<VkAttachmentDescription, 2> attachments = {};
    // Color attachment
    attachments[0].format = format.format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // Depth attachment
    attachments[1].format = depthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = {};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthReference = {};
    depthReference.attachment = 1;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.pDepthStencilAttachment = &depthReference;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = nullptr;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = nullptr;
    subpassDescription.pResolveAttachments = nullptr;

    // Subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(
        vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void VulkanWindow::tick() {
    // std::lock_guard<std::mutex> mtx_locker(sizeMtx);
    if (bCanDraw) {
        // 发送信号给presentComplete
        VkResult result = vkAcquireNextImageKHR(
            device, swapChain, UINT64_MAX, presentComplete, (VkFence) nullptr,
            &currentIndex);
        // 运行时提交,可能会导致一边在执行,一边在记录
        if (!this->bPreCmd && onCmdExecuteEvent) {
            result = vkWaitForFences(device, 1, &addCmdFences[currentIndex],
                                     true, UINT64_MAX);
            vkResetFences(device, 1, &addCmdFences[currentIndex]);
            vkBeginCommandBuffer(cmdBuffers[currentIndex], &cmdBufferBeginInfo);
            vkCmdSetViewport(cmdBuffers[currentIndex], 0, 1, &viewport);
            vkCmdSetScissor(cmdBuffers[currentIndex], 0, 1, &scissor);
            // renderPassBeginInfo.framebuffer = frameBuffers[currentIndex];
            // renderpass关联渲染目标 BlitImage不能包含在RenderPass里面
            // vkCmdBeginRenderPass(cmdBuffers[currentIndex],
            // &renderPassBeginInfo,
            //                      VK_SUBPASS_CONTENTS_INLINE);
            if (onCmdExecuteEvent) {
                onCmdExecuteEvent(currentIndex);
            }
            // vkCmdEndRenderPass(cmdBuffers[currentIndex]);
            vkEndCommandBuffer(cmdBuffers[currentIndex]);
        }
        // 提交缓冲区命令,执行完后发送信号给renderComplete
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffers[currentIndex];
        // 等presentComplete收到信号后执行,执行完成后发送信号renderComplete
        VkResult res = vkQueueSubmit(presentQueue, 1, &submitInfo,
                                     addCmdFences[currentIndex]);
        // 提交渲染呈现到屏幕
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &currentIndex;
        // 等待renderComplete
        presentInfo.pWaitSemaphores = &renderComplete;
        presentInfo.waitSemaphoreCount = 1;
        // 提交渲染呈现到屏幕
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            onSizeChange();
        } else {
            // assert(!result);
        }
        // // addCmdFence开门(发送信号量)
        // VK_CHECK_RESULT(
        //     vkQueueSubmit(presentQueue, 0, nullptr, addCmdFence));
    }
}

void VulkanWindow::run() {
    // 不是自己的窗口,不瞎BB
    if (!bCreateWindow) {
        return;
    }
#if defined(_WIN32)
    while (true) {
        bool quit = false;
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                quit = true;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (quit) {
            break;
        }
        tick();
    }

#elif defined(__ANDROID__)
    android_app* androidApp = AoceManager::Get().getApp();
    while (true) {
        int ident;
        int events;
        struct android_poll_source* source;
        bool destroy = false;

        focused = true;
        while ((ident = ALooper_pollAll(focused ? 0 : -1, NULL, &events,
                                        (void**)&source)) >= 0) {
            if (source != NULL) {
                source->process(androidApp, source);
            }
            if (androidApp->destroyRequested != 0) {
                LOGI("Android app destroy requested");
                destroy = true;
                break;
            }
        }
        // App destruction requested
        // Exit loop, example will be destroyed in application main
        if (destroy) {
            ANativeActivity_finish(androidApp->activity);
            break;
        }
        tick();
    }
#endif
}

void VulkanWindow::onSizeChange() {
    vkDeviceWaitIdle(device);
    createSwipChain(false);
    vkDeviceWaitIdle(device);
}

}  // namespace vulkan
}  // namespace aoce