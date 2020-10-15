#pragma once

#include <Layer/BaseLayer.hpp>

#include "../vulkan/VulkanContext.hpp"
#include "../vulkan/VulkanPipeline.hpp"
#include "VkHelper.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

typedef std::shared_ptr<VulkanBuffer> VulkanBufferPtr;
typedef std::shared_ptr<VulkanTexture> VulkanTexturePtr;

// 外部Vulkan层实现请继承这个类,提供对应计算图上的VulkanContext
class AOCE_VULKAN_EXPORT VkLayer : public BaseLayer {
    friend class VkPipeGraph;

   protected:
#if WIN32
    int32_t groupX = 32;
    int32_t groupY = 8;
#else
    int32_t groupX = 16;
    int32_t groupY = 16;
#endif
    int32_t groupZ = 1;
    // 默认提供一个参数buf
    std::unique_ptr<VulkanBuffer> constBuf;
    std::unique_ptr<UBOLayout> layout = nullptr;
    std::unique_ptr<VkPipeline> pipeLine = nullptr;

    GpuType gpu = GpuType::vulkan;
    class VkPipeGraph* vkPipeGraph = nullptr;
    VulkanContext* context = nullptr;

    std::vector<VulkanTexturePtr> inTexs;
    std::vector<VulkanTexturePtr> outTexs;

    VkCommandBuffer cmd;

   public:
    VkLayer(/* args */);
    ~VkLayer() override;

   public:
    virtual void onInit() final;
    // virtual void onInitLayer() override;
    virtual void onInitBuffer() final;
    virtual bool onFrame() override;

   public:
    // vulkan层在onInit后,shader 编译
    virtual void onInitPipe(){};
    // onInitBuffer后,onInitBuffer已经关联后上层的输出当做本层的输入
    virtual void onInitVkBuffer(){};
    virtual void onPreCmd(){};
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce