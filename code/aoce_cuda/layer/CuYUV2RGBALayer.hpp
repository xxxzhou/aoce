#pragma once
#include <Layer/InputLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuYUV2RGBALayer : public CuLayer, public YUV2RGBALayer {
    AOCE_LAYER_QUERYINTERFACE(CuYUV2RGBALayer)
   private:
    /* data */
   public:
    CuYUV2RGBALayer(/* args */);
    ~CuYUV2RGBALayer();

   protected:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace cuda
}  // namespace aoce