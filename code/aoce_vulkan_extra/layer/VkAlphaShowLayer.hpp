#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkAlphaShowLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkAlphaShowLayer)
   private:
    /* data */

   public:
    VkAlphaShowLayer();
    virtual ~VkAlphaShowLayer();

   protected:
    virtual void onInitLayer() override;
};

class VkAlphaShow2Layer : public VkLayer {
    AOCE_LAYER_GETNAME(VkAlphaShow2Layer)
   public:
    VkAlphaShow2Layer();
    virtual ~VkAlphaShow2Layer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce