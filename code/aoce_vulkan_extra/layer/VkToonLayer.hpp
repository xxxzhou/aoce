#pragma once

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkToonLayer : public VkLayer, public ITLayer<ToonParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkToonLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkToonLayer(/* args */);
    ~VkToonLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce