#pragma once

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkOperateLayer : public VkLayer, public ITexOperateLayer {
    AOCE_LAYER_QUERYINTERFACE(VkOperateLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkOperateLayer(/* args */);
    ~VkOperateLayer();

};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce