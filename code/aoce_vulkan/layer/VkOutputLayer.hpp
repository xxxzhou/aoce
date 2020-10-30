#pragma once
#include <Layer/OutputLayer.hpp>

#include "VkLayer.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

class VkOutputLayer : public OutputLayer, public VkLayer {
    AOCE_LAYER_QUERYINTERFACE(VkOutputLayer)
   private:
    // CPU输出使用
    std::unique_ptr<VulkanBuffer> outBuffer;
    std::vector<uint8_t> cpuData;
    std::unique_ptr<VulkanTexture> outTex;
    VkEvent outEvent = VK_NULL_HANDLE;

   public:
    VkOutputLayer(/* args */);
    ~VkOutputLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitVkBuffer() override;
    virtual void onPreCmd() override;
    virtual bool onFrame() override;

   public:
    virtual void outGpuTex(const VkOutGpuTex& outTex,
                           int32_t outIndex = 0) override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce