#include "VulkanModule.hpp"

#include <AoceCore.h>

namespace aoce {
namespace win {
namespace mf {
VulkanModule::VulkanModule(/* args */) {}

VulkanModule::~VulkanModule() {}

bool VulkanModule::loadModule() { return true; }

void VulkanModule::unloadModule() {}

ADD_MODULE(VulkanModule, aoce_vulkan)

}  // namespace mf
}  // namespace win
}  // namespace aoce