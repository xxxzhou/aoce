#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkCGAColorspaceLayer : public VkLayer {
   private:
    /* data */
   public:
    VkCGAColorspaceLayer(/* args */);
    virtual ~VkCGAColorspaceLayer();

   protected:
    virtual bool getSampled(int inIndex) override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce