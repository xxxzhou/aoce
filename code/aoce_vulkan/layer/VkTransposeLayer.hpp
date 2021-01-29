#pragma once

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkTransposeLayer : public VkLayer, public TransposeLayer {
    AOCE_LAYER_QUERYINTERFACE(VkTransposeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkTransposeLayer(/* args */);
    ~VkTransposeLayer();

   public:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce