#pragma once

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkMapChannelLayer : public VkLayer, public IMapChannelLayer {
    AOCE_LAYER_QUERYINTERFACE(VkMapChannelLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkMapChannelLayer();
    virtual ~VkMapChannelLayer();
};

class VkFlipLayer : public VkLayer, public IFlipLayer {
    AOCE_LAYER_QUERYINTERFACE(VkFlipLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkFlipLayer();
    virtual ~VkFlipLayer();
};

// class VkOperateLayer : public VkLayer, public ITexOperateLayer {
//     AOCE_LAYER_QUERYINTERFACE(VkOperateLayer)
//     AOCE_VULKAN_PARAMETUPDATE()
//    private:
//     /* data */
//    public:
//     VkOperateLayer(/* args */);
//     ~VkOperateLayer();
// };

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce