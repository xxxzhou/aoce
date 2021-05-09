#pragma once

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkPerlinNoiseLayer : public VkLayer, public PerlinNoiseLayer {
    AOCE_LAYER_QUERYINTERFACE(VkPerlinNoiseLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
    int32_t width = 0;
    int32_t height = 0;

   public:
    VkPerlinNoiseLayer(/* args */);
    ~VkPerlinNoiseLayer();

   public:
    virtual void setImageSize(int32_t width, int32_t height) override;
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce