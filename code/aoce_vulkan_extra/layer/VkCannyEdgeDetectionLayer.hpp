#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkLuminanceLayer.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkDirectionalSobelEdgeDetectionLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkDirectionalSobelEdgeDetectionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkDirectionalSobelEdgeDetectionLayer(/* args */);
    ~VkDirectionalSobelEdgeDetectionLayer();

   protected:
    virtual void onInitGraph() override;
};

struct DirectionalNMSParamet {
    float minThreshold;
    float maxThreshold;

    inline bool operator==(const DirectionalNMSParamet& right) {
        return this->minThreshold == right.minThreshold &&
               this->maxThreshold == right.maxThreshold;
    }
};

class VkDirectionalNMS : public VkLayer, public ITLayer<DirectionalNMSParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkDirectionalNMS)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkDirectionalNMS(/* args */);
    virtual ~VkDirectionalNMS();

   protected:
    virtual bool getSampled(int inIndex) override;
    virtual void onInitGraph() override;
};

class VkCannyEdgeDetectionLayer : public VkLayer,
                                  public ITLayer<CannyEdgeDetectionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkCannyEdgeDetectionLayer)
   private:
    /* data */
    std::unique_ptr<VkLuminanceLayer> luminanceLayer = nullptr;
    std::unique_ptr<VkGaussianBlurSLayer> gaussianBlurLayer = nullptr;
    std::unique_ptr<VkDirectionalSobelEdgeDetectionLayer> sobelEDLayer = nullptr;
    std::unique_ptr<VkDirectionalNMS> directNMSLayer = nullptr;

   public:
    VkCannyEdgeDetectionLayer(/* args */);
    virtual ~VkCannyEdgeDetectionLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce