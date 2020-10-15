#include "VulkanManager.hpp"
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

VulkanManager::~VulkanManager() {}

void VulkanManager::CreateInstance(const char* appName) {
#if __ANDROID__
    if (!InitVulkan()) {
        logMessage(LogLevel::error, "Failied initializing Vulkan APIs!");
        return false;
    }
#endif
    createInstance(instace, appName);
    std::vector<PhysicalDevice> physicalDevices;
    enumerateDevice(instace, physicalDevices);
    // 选择第一个物理设备
    this->physicalDevice = physicalDevices[0];
}

void VulkanManager::CreateDevice(uint32_t graphicsIndex, bool bAloneCompute) {
    // 创建虚拟设备
    createLogicalDevice(logicalDevice, physicalDevice, graphicsIndex,
                        bAloneCompute);
    vkGetDeviceQueue(logicalDevice.device, logicalDevice.computeIndex, 0,
                     &computeQueue);
    vkGetDeviceQueue(logicalDevice.device, logicalDevice.graphicsIndex, 0,
                     &graphicsQueue);
}

}  // namespace vulkan
}  // namespace aoce