#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkColorMatrixLayer : public VkLayer, public ITLayer<ColorMatrixParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkColorMatrixLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkColorMatrixLayer(/* args */);
    virtual ~VkColorMatrixLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce