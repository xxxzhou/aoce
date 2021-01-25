#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkChromKeyLayer : public VkLayer, public ITLayer<ChromKeyParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkChromKeyLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   protected:
   public:
    VkChromKeyLayer(/* args */);
    ~VkChromKeyLayer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce