#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkMedianLayer : public VkLayer, public ITLayer<uint32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkMedianLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
    bool bSingle = false;

   public:
    VkMedianLayer(bool bSingle = false);
    ~VkMedianLayer();

   protected:
    virtual void onInitGraph() override;
};

class VkMedianK3Layer : public VkLayer {
    AOCE_LAYER_GETNAME(VkMedianK3Layer)
   private:
    /* data */
    bool bSingle = false;

   public:
    VkMedianK3Layer(bool bSingle = false);
    ~VkMedianK3Layer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce