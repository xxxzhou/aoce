#include "VulkanShader.hpp"

#include "VulkanHelper.hpp"
#include "VulkanManager.hpp"

namespace aoce {
namespace vulkan {

VulkanShader::VulkanShader(/* args */) {}

VulkanShader::~VulkanShader() { release(); }

void VulkanShader::release() {
    if (shaderModule) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }
}

void VulkanShader::loadShaderModule(VkDevice device, std::string path,
                                    VkShaderStageFlagBits shaderFlag) {
    release();
#if defined(__ANDROID__)
    AAssetManager* assetManager =
        VulkanManager::Get().androidApp->activity->assetManager;
    shaderModule = loadShader(assetManager, path.c_str(), device);
#else
    shaderModule = loadShader(path.c_str(), device);
#endif
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = shaderFlag;
    shaderStage.pName = "main";
    shaderStage.module = shaderModule;
}

}  // namespace vulkan
}  // namespace aoce