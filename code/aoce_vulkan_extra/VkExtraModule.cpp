#include "VkExtraModule.hpp"

namespace aoce {
namespace vulkan {

VkExtraModule::VkExtraModule() {}

VkExtraModule::~VkExtraModule() {}

bool VkExtraModule::loadModule() {
    return true;
}

void VkExtraModule::unloadModule() {}

}
}