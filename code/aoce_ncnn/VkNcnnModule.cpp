#include "VkNcnnModule.hpp"

#include "CNNHelper.hpp"
#include "aoce_vulkan/vulkan/VulkanHelper.hpp"
#include "net.h"
namespace aoce {
// 是否使用ncnn的vulkan上下文替换aoce_vulkan本身的
#define NCNN_VULKAN_REPLACE 0
// 使用的ncnn版本是否放出VkInstance并且包含VK_KHR_WIN32_SURFACE_EXTENSION_NAME
#define NCNN_WIN32_VULKAN_INSTANCE 0

static NcnnGlobeParamet ngParamet = {};

const NcnnGlobeParamet& getNgParamet() { return ngParamet; }

VkNcnnModule::VkNcnnModule(/* args */) {}

VkNcnnModule::~VkNcnnModule() {}

bool VkNcnnModule::loadModule() {
    // ncnn::create_gpu_instance();
    int count = ncnn::get_gpu_count();
    if (count > 0) {
        const ncnn::GpuInfo& gpuInfo = ncnn::get_gpu_info();
        ncnn::VulkanDevice* device = ncnn::get_gpu_device();
        ngParamet.vkDevice = device;
        ngParamet.vkAllocator = device->acquire_staging_allocator();
#if NCNN_VULKAN_REPLACE && WIN32
        // aoce_ncnn vulkan context replace aoce_vulkan
        logMessage(aoce::LogLevel::info,
                   "use aoce_ncnn vulkan context replace aoce_vulkan.");
        ngParamet.bOneVkDevice = true;

#if NCNN_WIN32_VULKAN_INSTANCE
        aoce::vulkan::setVulkanContext(gpuInfo.physical_device(),
                                       device->vkdevice(), gpuInfo.instance());
#else
        // android部分,ncnn的vulkan没有VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
        // 导致不能把vkImage复制给opengl纹理
        aoce::vulkan::setVulkanContext(gpuInfo.physical_device(),
                                       device->vkdevice());
#endif
#endif
    }
    return true;
}

void VkNcnnModule::unloadModule() {
    if (ngParamet.vkAllocator) {
        ngParamet.vkDevice->reclaim_staging_allocator(ngParamet.vkAllocator);
        ngParamet.vkAllocator = nullptr;
        ngParamet.vkDevice = nullptr;
        ngParamet.bOneVkDevice = false;
    }
    // ncnn::destroy_gpu_instance();
}

ADD_MODULE(VkNcnnModule, aoce_ncnn)

}  // namespace aoce