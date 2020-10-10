#include "VulkanModule.hpp"

#include <AoceCore.h>

#include "layer/VkLayerFactory.hpp"
#include "layer/VkPipeGraph.hpp"

#if __ANDROID__
#include "../android/vulkan_wrapper.h"
#endif

namespace aoce {
namespace vk {

VulkanModule::VulkanModule(/* args */) {}

VulkanModule::~VulkanModule() {}

bool VulkanModule::loadModule() {
#if __ANDROID__
    if (!InitVulkan()) {
        logMessage(LogLevel::error, "Failied initializing Vulkan APIs!");
        return false;
    }
#endif
    AoceManager::Get().addPipeGraphFactory(GpuType::vulkan,
                                           new layer::VkPipeGraphFactory());
    AoceManager::Get().addLayerFactory(GpuType::vulkan,
                                       new layer::VkLayerFactory());
    return true;
}

void VulkanModule::unloadModule() {
    AoceManager::Get().removePipeGraphFactory(GpuType::vulkan);
    AoceManager::Get().removeLayerFactory(GpuType::vulkan);
}

ADD_MODULE(VulkanModule, aoce_vulkan)

}  // namespace vk
}  // namespace aoce