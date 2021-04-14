#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkColorBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorBlendLayer(/* args */);
    ~VkColorBlendLayer();
};

class VkColorBurnBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorBurnBlendLayer(/* args */);
    ~VkColorBurnBlendLayer();
};

class VkColorDodgeBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorDodgeBlendLayer(/* args */);
    ~VkColorDodgeBlendLayer();
};

class VkColorInvertLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorInvertLayer(/* args */);
    ~VkColorInvertLayer();
};

class VkColorLBPLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorLBPLayer(/* args */);
    ~VkColorLBPLayer();
};

class VkContrastLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkContrastLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkContrastLayer(/* args */);
    ~VkContrastLayer();
};

class VkCrosshatchLayer : public VkLayer, public ITLayer<CrosshatchParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkCrosshatchLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkCrosshatchLayer(/* args */);
    ~VkCrosshatchLayer();
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce