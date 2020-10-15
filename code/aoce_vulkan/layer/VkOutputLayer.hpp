#pragma once
#include <Layer/OutputLayer.hpp>

#include "VkLayer.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

class VkOutputLayer : public OutputLayer, public VkLayer {
    AOCE_LAYER_QUERYINTERFACE(VkOutputLayer)
   private:
    std::unique_ptr<VulkanBuffer> outBuffer;
    std::vector<uint8_t> cpuData;

   public:
    VkOutputLayer(/* args */);
    ~VkOutputLayer();

   public:
    virtual void onInitVkBuffer() override;
    virtual void onPreCmd() override;
    virtual bool onFrame() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce