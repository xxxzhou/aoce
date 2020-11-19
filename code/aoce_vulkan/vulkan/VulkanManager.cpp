
#include "VulkanManager.hpp"

#include <vulkan/vulkan.h>
#if __ANDROID__
#include "../android/vulkan_wrapper.h"
#endif

namespace aoce {
namespace vulkan {

VulkanManager* VulkanManager::instance = nullptr;
VulkanManager::VulkanManager(/* args */) {}

VulkanManager& VulkanManager::Get() {
    if (instance == nullptr) {
        instance = new VulkanManager();
    }
    return *instance;
}

VulkanManager::~VulkanManager() {
    if (device) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (instace) {
        vkDestroyInstance(instace, nullptr);
        instace = VK_NULL_HANDLE;
    }
    physicalDevice = VK_NULL_HANDLE;
}

bool VulkanManager::createInstance(const char* appName) {
#if __ANDROID__
    if (!InitVulkan()) {
        logMessage(LogLevel::error, "Failied initializing Vulkan APIs!");
        return false;
    }
#endif
    aoce::vulkan::createInstance(instace, appName);
    std::vector<PhysicalDevicePtr> physicalDevices;
    enumerateDevice(instace, physicalDevices);
    if (physicalDevices.empty()) {
        return false;
    }
    bool find = false;
    // 首选独立显卡
    for (auto& pdevice : physicalDevices) {
        if (pdevice->properties.deviceType ==
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            this->physical = pdevice;
            find = true;
            break;
        }
    }
    if (!find) {
        physical = physicalDevices[0];
    }
    assert(physical != nullptr);

    std::string message = physical->properties.deviceName;
    logMessage(aoce::LogLevel::info, "select gpu: " + message);
    physicalDevice = this->physical->physicalDevice;
#if __ANDROID__
    uint32_t count = 0;
    std::vector<VkExtensionProperties> extensions;
    VkResult result = vkEnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &count, nullptr);
    if (result == VK_SUCCESS) {
        extensions.resize(count);
        result = vkEnumerateDeviceExtensionProperties(
            physicalDevice, nullptr, &count, extensions.data());
    }
    for (auto& extPorperty : extensions) {
        if (strcmp(
                extPorperty.extensionName,
                VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME) ==
            0) {
            bAndroidHardware = true;
            break;
        }
    }
#endif
    return true;
}

bool VulkanManager::findAloneCompute(int32_t& familyIndex) {
    const auto& queueFamilys = physical->queueGraphicsIndexs;
    for (const auto& cindex : physical->queueComputeIndexs) {
        if (std::find(queueFamilys.begin(), queueFamilys.end(), cindex) ==
            queueFamilys.end()) {
            familyIndex = cindex;
            return true;
        }
    }
    return false;
}

void VulkanManager::createDevice(bool bAloneCompute) {
    assert(physical->queueGraphicsIndexs.size() > 0);
    // 创建虚拟设备
    // 创建一个device,这个device根据条件能否访问graphics/compute
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    graphicsIndex = physical->queueGraphicsIndexs[0];
    computeIndex = graphicsIndex;
    if (bAloneCompute) {
        findAloneCompute(computeIndex);
    }
    float queuePriorities[1] = {0.0};
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = graphicsIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = queuePriorities;
    queueCreateInfos.push_back(queueInfo);
    if (computeIndex != graphicsIndex) {
        queueInfo = {};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = computeIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = queuePriorities;
        queueCreateInfos.push_back(queueInfo);
    }
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = nullptr;
    std::vector<const char*> deviceExtensions;
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if __ANDROID__
    // 和android里的AHardwareBuffer交互,没有的话,相关vkGetDeviceProcAddr获取不到对应函数
    deviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
    deviceExtensions.push_back(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
#endif
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    VK_CHECK_RESULT(
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));
    bAloneCompute = computeIndex != graphicsIndex;
    vkGetDeviceQueue(device, computeIndex, 0, &computeQueue);
    vkGetDeviceQueue(device, graphicsIndex, 0, &graphicsQueue);
}

bool VulkanManager::findSurfaceQueue(VkSurfaceKHR surface,
                                     int32_t& presentIndex) {
    uint32_t queueFamilyCount = physical->queueFamilyProps.size();
    std::vector<VkBool32> supportsPresent(queueFamilyCount);
    // 检查每个通道的表面是否支持显示
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface,
                                             &supportsPresent[i]);
    }
    // 查看是否能呈现渲染画面的
    if (supportsPresent[graphicsIndex] == VK_TRUE) {
        presentIndex = graphicsIndex;
        return true;
    }
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (supportsPresent[i] == VK_TRUE) {
            presentIndex = i;
            return false;
        }
    }
    presentIndex = -1;
    return false;
}

void VulkanManager::blitFillImage(VkCommandBuffer cmd, const VulkanTexture* src,
                                  VkImage dest, int32_t destWidth,
                                  int32_t destHeight,
                                  VkImageLayout destLayout) {
    VkImageBlit region = {};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.mipLevel = 0;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount = 1;
    region.srcOffsets[0] = {0, 0, 0};
    region.srcOffsets[1] = {(int32_t)src->width, (int32_t)src->height, 1};
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.mipLevel = 0;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount = 1;
    region.dstOffsets[0] = {0, 0, 0};
    region.dstOffsets[1] = {destWidth, destHeight, 1};
    // 要将源图像的区域复制到目标图像中，并可能执行格式转换，任意缩放和过滤
    vkCmdBlitImage(cmd, src->image, src->layout, dest, destLayout, 1, &region,
                   VK_FILTER_LINEAR);
}

void VulkanManager::copyImage(VkCommandBuffer cmd, const VulkanTexture* src,
                              const VulkanTexture* dest) {
    VkImageCopy copyRegion = {};

    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcOffset = {0, 0, 0};

    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.baseArrayLayer = 0;
    copyRegion.dstSubresource.mipLevel = 0;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.dstOffset = {0, 0, 0};

    copyRegion.extent.width = src->width;
    copyRegion.extent.height = src->height;
    copyRegion.extent.depth = 1;

    vkCmdCopyImage(cmd, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dest->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &copyRegion);
}

}  // namespace vulkan
}  // namespace aoce