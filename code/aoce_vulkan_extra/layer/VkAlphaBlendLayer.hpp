#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkAddBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkAddBlendLayer(/* args */);
    virtual ~VkAddBlendLayer();
};

class VkAlphaBlendLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkAlphaBlendLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkAlphaBlendLayer(/* args */);
    virtual ~VkAlphaBlendLayer();
};

class VkHardLightBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkHardLightBlendLayer(/* args */);
    ~VkHardLightBlendLayer();
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce