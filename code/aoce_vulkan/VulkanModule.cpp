#include "VulkanModule.hpp"



#include <AoceManager.hpp>

#include "VkLayerFactory.hpp"
#include "layer/VkPipeGraph.hpp"
#include "vulkan/VulkanManager.hpp"

namespace aoce {
namespace vulkan {

VulkanModule::VulkanModule(/* args */) {}

VulkanModule::~VulkanModule() {}

bool VulkanModule::loadModule() {
    bool bfindGpu = VulkanManager::Get().createInstance("aoce_vulkan");
    if (!bfindGpu) {
        logMessage(LogLevel::warn, "vulkan not find gpu.");
        return false;
    }
    logMessage(LogLevel::info, "add vulkan factory.");
    VulkanManager::Get().createDevice(true);
    AoceManager::Get().addPipeGraphFactory(GpuType::vulkan,
                                           new layer::VkPipeGraphFactory());
    AoceManager::Get().addLayerFactory(GpuType::vulkan,
                                       new layer::VkLayerFactory());
    return true;
}

void VulkanModule::unloadModule() {
    logMessage(LogLevel::info, "remove vulkan factory.");
    AoceManager::Get().removePipeGraphFactory(GpuType::vulkan);
    AoceManager::Get().removeLayerFactory(GpuType::vulkan);
}

ADD_MODULE(VulkanModule, aoce_vulkan)

}  // namespace vulkan
}  // namespace aoce