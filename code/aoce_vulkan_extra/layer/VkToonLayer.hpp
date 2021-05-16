#pragma once

#include "../VkExtraExport.h"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkToonLayer : public VkLayer, public ITLayer<ToonParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkToonLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkToonLayer(/* args */);
    ~VkToonLayer();
};

class VkSmoothToonLayer : public GroupLayer, public ITLayer<SmoothToonParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSmoothToonLayer)
   private:
    /* data */
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer = nullptr;
    std::unique_ptr<VkToonLayer> toonLayer = nullptr;

   public:
    VkSmoothToonLayer(/* args */);
    ~VkSmoothToonLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce