#pragma once

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkLuminanceLayer : public VkLayer {
   private:
    /* data */
   public:
    VkLuminanceLayer(/* args */);
    ~VkLuminanceLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
