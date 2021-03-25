#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkAlphaShowLayer : public VkLayer {
   private:
    /* data */

   public:
    VkAlphaShowLayer();
    ~VkAlphaShowLayer();

   protected:
    virtual void onInitLayer() override;
};

class VkAlphaShow2Layer : public VkLayer {
   public:
    VkAlphaShow2Layer();
    ~VkAlphaShow2Layer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce