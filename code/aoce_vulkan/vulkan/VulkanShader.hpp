#pragma once

#include <map>
#include <string>

#include "VulkanCommon.hpp"
namespace aoce {
namespace vulkan {

class AOCE_VULKAN_EXPORT VulkanShader {
   private:
    /* data */
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;

   public:
    VulkanShader(/* args */);
    ~VulkanShader();

   public:
    VkPipelineShaderStageCreateInfo shaderStage = {};

   private:
    void release();

   public:
    void loadShaderModule(
        VkDevice device, std::string path,
        VkShaderStageFlagBits shaderFlag = VK_SHADER_STAGE_COMPUTE_BIT);
};

}  // namespace vulkan
}  // namespace aoce