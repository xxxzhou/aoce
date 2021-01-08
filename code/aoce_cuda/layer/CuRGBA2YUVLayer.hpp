#pragma once

#include <Layer/BaseLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuRGBA2YUVLayer : public CuLayer, public RGBA2YUVLayer {
    AOCE_LAYER_QUERYINTERFACE(CuRGBA2YUVLayer)
   private:
    /* data */
   public:
    CuRGBA2YUVLayer(/* args */);
    ~CuRGBA2YUVLayer();

   protected:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace cuda
}  // namespace aoce