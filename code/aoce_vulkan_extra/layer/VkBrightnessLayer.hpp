#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBrightnessLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkBrightnessLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkBrightnessLayer(/* args */);
    virtual ~VkBrightnessLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce