#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
    
class VkSharpenLayer : public VkLayer, public ITLayer<SharpenParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSharpenLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSharpenLayer(/* args */);
    ~VkSharpenLayer();
};

class VkColorLBPLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkColorLBPLayer)
   private:
    /* data */
   public:
    VkColorLBPLayer(/* args */);
    virtual ~VkColorLBPLayer();
};


}  // namespace layer
}  // namespace vulkan
}  // namespace aoce