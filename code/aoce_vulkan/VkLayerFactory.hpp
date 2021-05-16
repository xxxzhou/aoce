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
    virtual IInputLayer* crateInput() override;
    virtual IOutputLayer* createOutput() override;
    virtual IYUV2RGBALayer* createYUV2RGBA() override;
    virtual IRGBA2YUVLayer* createRGBA2YUV() override;
    virtual ITexOperateLayer* createTexOperate() override;
    virtual ITransposeLayer* createTranspose() override;
    virtual IReSizeLayer* createSize() override;
    virtual IBlendLayer* createBlend() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce