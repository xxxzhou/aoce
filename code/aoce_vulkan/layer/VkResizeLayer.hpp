#pragma once
#include <Layer/BaseLayer.hpp>
#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkResizeLayer : public VkLayer, public ReSizeLayer {
    AOCE_LAYER_QUERYINTERFACE(VkResizeLayer)   
   private:
    /* data */
   public:
    VkResizeLayer(/* args */);
    ~VkResizeLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;

    virtual void onInitPipe() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce