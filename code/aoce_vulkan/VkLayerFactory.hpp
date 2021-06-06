#pragma once

#include "aoce/AoceCore.h"

namespace aoce {
namespace vulkan {
namespace layer {

class VkLayerFactory : public LayerFactory {
   private:
    /* data */
   public:
    VkLayerFactory(/* args */);
    virtual ~VkLayerFactory() override;

   public:
    virtual IInputLayer* createInput() override;
    virtual IOutputLayer* createOutput() override;
    virtual IYUVLayer* createYUV2RGBA() override;
    virtual IYUVLayer* createRGBA2YUV() override;
    virtual IMapChannelLayer* createMapChannel() override;
    virtual IFlipLayer* createFlip() override;
    virtual ITransposeLayer* createTranspose() override;
    virtual IReSizeLayer* createSize() override;
    virtual IBlendLayer* createBlend() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce