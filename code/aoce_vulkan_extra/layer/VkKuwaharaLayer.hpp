#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 半径 1-32,默认为5
class VkKuwaharaLayer : public VkLayer, public ITLayer<uint32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkKuwaharaLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkKuwaharaLayer(/* args */);
    ~VkKuwaharaLayer();
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce