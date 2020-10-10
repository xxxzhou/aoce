#pragma once
#include <Layer/InputLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vk {
namespace layer {

class VkInputLayer : public InputLayer, public VkLayer {
   private:
    /* data */
   public:
    VkInputLayer(/* args */);
    ~VkInputLayer();

    // InputLayer
   public:
    virtual void setImage(ImageFormat imageFormat, int32_t index = 0) override;
    AOCE_LAYER_QUERYINTERFACE(VkInputLayer)
    // VkLayer
   public:
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vk
}  // namespace aoce