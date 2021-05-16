#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkColorAdjustmentLayer.hpp"
#include "VkLuminanceLayer.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"
#include "aoce_vulkan/layer/VkResizeLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkIOSBlurLayer : public GroupLayer, public ITLayer<IOSBlurParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkIOSBlurLayer)
   private:
    /* data */
   public:
    VkIOSBlurLayer(/* args */);
    ~VkIOSBlurLayer();

   private:
    std::unique_ptr<VkSizeScaleLayer> downLayer = nullptr;
    std::unique_ptr<VkSaturationLayer> saturationLayer = nullptr;
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer = nullptr;
    std::unique_ptr<VkLuminanceRangeLayer> lumRangeLayer = nullptr;
    std::unique_ptr<VkSizeScaleLayer> upLayer = nullptr;

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce