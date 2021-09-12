#include "VkNcnnModule.hpp"

#include "aoce_vulkan/vulkan/VulkanHelper.hpp"
#include "net.h"
namespace aoce {

VkNcnnModule::VkNcnnModule(/* args */) {}

VkNcnnModule::~VkNcnnModule() {}

bool VkNcnnModule::loadModule() {
    // ncnn::create_gpu_instance();
    int count = ncnn::get_gpu_count();
    if (count > 0) {
        // const ncnn::GpuInfo& gpuInfo = ncnn::get_gpu_info();
        // ncnn::VulkanDevice* device = ncnn::get_gpu_device();
        // aoce::vulkan::setVulkanContext(gpuInfo.physical_device(),
        //                                device->vkdevice());
    }
    return true;
}

void VkNcnnModule::unloadModule() { ncnn::destroy_gpu_instance(); }

ADD_MODULE(VkNcnnModule, aoce_ncnn)

}  // namespace aoce