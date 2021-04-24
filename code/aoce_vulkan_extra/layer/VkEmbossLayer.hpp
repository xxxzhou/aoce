#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// The strength of the embossing, from  0.0 to 4.0, with 1.0 as the normal level
class VkEmbossLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkEmbossLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkEmbossLayer(/* args */);
    ~VkEmbossLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce