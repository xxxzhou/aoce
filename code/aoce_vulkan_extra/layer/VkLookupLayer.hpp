#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkLookupLayer : public VkLayer {
   private:
    /* data */
   public:
    VkLookupLayer(/* args */);
    virtual ~VkLookupLayer();

   protected:
    virtual bool getSampled(int inIndex) override;
    virtual bool sampledNearest(int32_t inIndex) override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce