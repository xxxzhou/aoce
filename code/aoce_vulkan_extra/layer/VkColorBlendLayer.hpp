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
    virtual ~VkColorBlendLayer();
};

class VkColorBurnBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorBurnBlendLayer(/* args */);
    virtual ~VkColorBurnBlendLayer();
};

class VkColorDodgeBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorDodgeBlendLayer(/* args */);
    virtual ~VkColorDodgeBlendLayer();
};

class VkColorInvertLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorInvertLayer(/* args */);
    virtual ~VkColorInvertLayer();
};

class VkColorLBPLayer : public VkLayer {
   private:
    /* data */
   public:
    VkColorLBPLayer(/* args */);
    virtual ~VkColorLBPLayer();
};

class VkContrastLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkContrastLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkContrastLayer(/* args */);
    virtual ~VkContrastLayer();
};

class VkCrosshatchLayer : public VkLayer, public ITLayer<CrosshatchParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkCrosshatchLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkCrosshatchLayer(/* args */);
    virtual ~VkCrosshatchLayer();
};

class VkExclusionBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkExclusionBlendLayer(/* args */);
    ~VkExclusionBlendLayer();
};

class VkFalseColorLayer : public VkLayer, public ITLayer<FalseColorParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkFalseColorLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkFalseColorLayer(/* args */);
    ~VkFalseColorLayer();
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce