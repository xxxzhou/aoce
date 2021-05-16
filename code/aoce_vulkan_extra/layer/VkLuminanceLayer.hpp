#pragma once

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkLuminanceLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkLuminanceLayer)
   private:
    /* data */
   public:
    VkLuminanceLayer(/* args */);
    ~VkLuminanceLayer();

   protected:
    virtual void onInitGraph() override;
};

// 降低亮度范围的程度,从0.0到1.0. 默认值为0.6.
class VkLuminanceRangeLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkLuminanceRangeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkLuminanceRangeLayer(/* args */);
    ~VkLuminanceRangeLayer();
};

class VkLuminanceThresholdLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkLuminanceThresholdLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkLuminanceThresholdLayer(/* args */);
    ~VkLuminanceThresholdLayer();

   protected:
    virtual void onInitGraph() override;
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
