#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkCropLayer : public VkLayer, public ITLayer<CropParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkCropLayer)
   private:
    /* data */
   public:
    VkCropLayer(/* args */);
    ~VkCropLayer();

   private:
    bool parametTransform();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce