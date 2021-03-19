#include "VulkanShader.hpp"

#include <AoceManager.hpp>

#include "VulkanHelper.hpp"

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
    this->device = device;
    release();
#if defined(__ANDROID__)
    AAssetManager* assetManager = AoceManager::Get().getAppEnv().assetManager;
    assert(assetManager != nullptr);
    shaderModule = loadShader(assetManager, path.c_str(), device);
#else
    std::string fullPath = getAocePath() + "/" + path;
    shaderModule = loadShader(fullPath.c_str(), device);
#endif
    logAssert(shaderModule != VK_NULL_HANDLE, "file: " + path + " load shader failed");
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = shaderFlag;
    shaderStage.pName = "main";
    shaderStage.module = shaderModule;
}

}  // namespace vulkan
}  // namespace aoce