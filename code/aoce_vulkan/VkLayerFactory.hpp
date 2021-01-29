#pragma once

#include <Layer/LayerFactory.hpp>

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
    virtual InputLayer* crateInput() override;
    virtual OutputLayer* createOutput() override;
    virtual YUV2RGBALayer* createYUV2RGBA() override;
    virtual RGBA2YUVLayer* createRGBA2YUV() override;
    virtual TexOperateLayer* createTexOperate() override;
    virtual TransposeLayer* createTranspose() override;
    virtual ReSizeLayer* createSize() override;
    virtual BlendLayer* createBlend() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce