#pragma once
#include <Layer/BaseLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
class VkRGBA2YUVLayer : public VkLayer, public RGBA2YUVLayer {
    AOCE_LAYER_QUERYINTERFACE(VkRGBA2YUVLayer)
   private:
    /* data */
   public:
    VkRGBA2YUVLayer(/* args */);
    ~VkRGBA2YUVLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
    virtual void onInitPipe() override;
    virtual void onPreCmd() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce