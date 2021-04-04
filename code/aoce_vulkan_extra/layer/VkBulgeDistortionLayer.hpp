#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBulgeDistortionLayer : public VkLayer,
                               public ITLayer<BulgeDistortionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkBulgeDistortionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkBulgeDistortionLayer(/* args */);
    virtual ~VkBulgeDistortionLayer();

   protected:
    virtual bool getSampled(int inIndex) override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce