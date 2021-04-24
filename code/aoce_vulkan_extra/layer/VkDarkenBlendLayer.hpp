#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkDarkenBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkDarkenBlendLayer(/* args */);
    virtual ~VkDarkenBlendLayer();
};

class VkDifferenceBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkDifferenceBlendLayer(/* args */);
    virtual ~VkDifferenceBlendLayer();
};

class VkDissolveBlendLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkDissolveBlendLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkDissolveBlendLayer(/* args */);
    virtual ~VkDissolveBlendLayer();
};

class VkDivideBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkDivideBlendLayer(/* args */);
    virtual ~VkDivideBlendLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce