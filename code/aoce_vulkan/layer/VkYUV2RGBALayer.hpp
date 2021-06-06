#pragma once
#include <layer/BaseLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
class VkYUV2RGBALayer : public VkLayer, public IYUVLayer {
    AOCE_LAYER_QUERYINTERFACE(VkYUV2RGBALayer)
   private:
    /* data */
   public:
    VkYUV2RGBALayer(/* args */);
    ~VkYUV2RGBALayer();

   protected:
    
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override; 
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce