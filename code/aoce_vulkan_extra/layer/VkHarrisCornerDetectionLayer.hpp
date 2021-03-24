#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkXYDerivativeLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkXYDerivativeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkXYDerivativeLayer();
    virtual ~VkXYDerivativeLayer();

   protected:
    virtual void onInitGraph() override;
};

// GPUImageThresholdedNonMaximumSuppressionFilter
class VkThresholdedNMS : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkThresholdedNMS)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkThresholdedNMS();
    virtual ~VkThresholdedNMS();

   protected:
    virtual void onInitGraph() override;
};

class VkHarrisCornerDetectionLayer
    : public VkLayer,
      public ITLayer<HarrisCornerDetectionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkHarrisCornerDetectionLayer)
   private:
    /* data */
    std::unique_ptr<VkXYDerivativeLayer> xyDerivativeLayer;
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer;
    std::unique_ptr<VkThresholdedNMS> thresholdNMSLayer;

   public:
    VkHarrisCornerDetectionLayer(/* args */);
    virtual ~VkHarrisCornerDetectionLayer();

   private:
    void updateUBO();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce