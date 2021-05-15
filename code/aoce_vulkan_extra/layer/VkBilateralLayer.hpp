#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 双边滤波
class VkBilateralLayer : public VkLayer, public ITLayer<BilateralParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkBilateralLayer)
   private:
    /* data */
   public:
    VkBilateralLayer(/* args */);
    virtual ~VkBilateralLayer();

   private:
    void transformParamet();

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce