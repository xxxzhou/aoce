#include "VulkanModule.hpp"

#include <AoceCore.h>

#include "VkLayerFactory.hpp"
#include "layer/VkPipeGraph.hpp"
#include "vulkan/VulkanManager.hpp"

#if __ANDROID__
#include "../android/vulkan_wrapper.h"
#endif

namespace aoce {
namespace vulkan {

VulkanModule::VulkanModule(/* args */) {}

VulkanModule::~VulkanModule() {}

bool VulkanModule::loadModule() {
    VulkanManager::Get().CreateInstance("aoce_vulkan");
    uint32_t graphicIndex =
        VulkanManager::Get().physicalDevice.queueGraphicsIndexs[0];
    VulkanManager::Get().CreateDevice(graphicIndex, true);
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

}  // namespace vulkan
}  // namespace aoce