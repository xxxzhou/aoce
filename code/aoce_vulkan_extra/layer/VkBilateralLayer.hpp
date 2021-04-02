#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBilateralLayer : public VkLayer, public ITLayer<BilateralParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkBilateralLayer)
   private:
    /* data */
   public:
    VkBilateralLayer(/* args */);
    ~VkBilateralLayer();

   private:
    void transformParamet();

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce