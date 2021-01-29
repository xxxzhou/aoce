#pragma once

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkOperateLayer : public VkLayer, public TexOperateLayer {
    AOCE_LAYER_QUERYINTERFACE(VkOperateLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkOperateLayer(/* args */);
    ~VkOperateLayer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce