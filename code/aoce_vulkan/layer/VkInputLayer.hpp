#pragma once
#include <Layer/InputLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 把各种VideoFormat转化成ImageFormat,主要二种,R8/RGBA8
class VkInputLayer : public InputLayer, public VkLayer {
    AOCE_LAYER_QUERYINTERFACE(VkInputLayer)
   private:
    VideoFormat videoFormat;
    std::unique_ptr<VulkanBuffer> inBuffer;
    uint8_t* frameData = nullptr;

   public:
    VkInputLayer(/* args */);
    ~VkInputLayer();

    // InputLayer
   protected:
    virtual void onSetImage(VideoFormat videoFormat,
                            int32_t index = 0) override;
    virtual void onInputCpuData(uint8_t* data, int32_t index = 0) override;
    // VkLayer
   public:
    virtual void onInitVkBuffer() override;
    virtual void onPreCmd() override;
    virtual bool onFrame() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce