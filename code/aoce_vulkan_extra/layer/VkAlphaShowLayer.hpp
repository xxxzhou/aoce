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
    VkAlphaShowLayer(/* args */);
    ~VkAlphaShowLayer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce