#pragma once
#include <Layer/BaseLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
class VkYUV2RGBALayer : public VkLayer, public YUV2RGBALayer {
    AOCE_LAYER_QUERYINTERFACE(VkYUV2RGBALayer)
   private:
    /* data */
   public:
    VkYUV2RGBALayer(/* args */);
    ~VkYUV2RGBALayer();

   public:
    virtual void onInitGraph() override;
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
    virtual void onInitPipe() override;
    virtual void onPreCmd() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce