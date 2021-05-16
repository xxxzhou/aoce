#pragma once

#include <memory>

#include "../VkExtraExport.h"
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

class VkHarrisDetectionBaseLayer : public VkLayer {
   protected:
    /* data */
    std::unique_ptr<VkXYDerivativeLayer> xyDerivativeLayer;
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer;
    std::unique_ptr<VkThresholdedNMS> thresholdNMSLayer;

   public:
    VkHarrisDetectionBaseLayer();
    virtual ~VkHarrisDetectionBaseLayer();

   protected:
    void baseParametChange(const HarrisDetectionBaseParamet& baseParamet);

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

class VkHarrisCornerDetectionLayer
    : public VkHarrisDetectionBaseLayer,
      public ITLayer<HarrisCornerDetectionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkHarrisCornerDetectionLayer)
   public:
    VkHarrisCornerDetectionLayer(/* args */);
    virtual ~VkHarrisCornerDetectionLayer();

   protected:
    void transformParamet();
    virtual void onUpdateParamet() override;
};

class VkNobleCornerDetectionLayer
    : public VkHarrisDetectionBaseLayer,
      public ITLayer<NobleCornerDetectionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkNobleCornerDetectionLayer)
   private:
    /* data */
   public:
    VkNobleCornerDetectionLayer(/* args */);
    virtual ~VkNobleCornerDetectionLayer();

   protected:
    void transformParamet();
    virtual void onUpdateParamet() override;
};

class VkShiTomasiFeatureDetectionLayer : public VkNobleCornerDetectionLayer {
    AOCE_LAYER_QUERYINTERFACE(VkShiTomasiFeatureDetectionLayer)
   private:
    /* data */
   public:
    VkShiTomasiFeatureDetectionLayer(/* args */);
    virtual ~VkShiTomasiFeatureDetectionLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce