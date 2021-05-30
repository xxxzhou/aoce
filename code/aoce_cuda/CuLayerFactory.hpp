#pragma once

#include "aoce/AoceCore.h"

namespace aoce {
namespace cuda {

class CuLayerFactory : public LayerFactory {
   private:
    /* data */
   public:
    CuLayerFactory(/* args */);
    virtual ~CuLayerFactory() override;

   public:
    virtual IInputLayer* crateInput() override;
    virtual IOutputLayer* createOutput() override;
    virtual IYUV2RGBALayer* createYUV2RGBA() override;
    virtual IRGBA2YUVLayer* createRGBA2YUV() override;
    virtual IMapChannelLayer* createMapChannel() override;
    virtual IFlipLayer* createFlip() override;
    virtual ITransposeLayer* createTranspose() override;
    virtual IReSizeLayer* createSize() override;
    virtual IBlendLayer* createBlend() override;
};

}  // namespace cuda
}  // namespace aoce
