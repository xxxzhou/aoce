#pragma once

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkTransposeLayer : public VkLayer, public ITransposeLayer {
    AOCE_LAYER_QUERYINTERFACE(VkTransposeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkTransposeLayer(/* args */);
    ~VkTransposeLayer();

   public:    
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce