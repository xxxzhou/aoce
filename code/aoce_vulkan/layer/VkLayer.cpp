#include "VkLayer.hpp"

#include "VkPipeGraph.hpp"

#ifdef _WIN32
#include <windows.h>
#elif __ANDROID__
#include <android/asset_manager.h>
#endif

namespace aoce {
namespace vk {
namespace layer {

VkLayer::VkLayer(/* args */) { gpu = GpuType::vulkan; }

VkLayer::~VkLayer() {}

void VkLayer::onInit() {
    vkPipeGraph =
        static_cast<VkPipeGraph*>(pipeGraph);  // dynamic_cast android rtti
    // assert(vkPipeGraph != nullptr);
}

void VkLayer::onInitLayer() {}

void VkLayer::onRun() {}

}  // namespace layer
}  // namespace vk
}  // namespace aoce