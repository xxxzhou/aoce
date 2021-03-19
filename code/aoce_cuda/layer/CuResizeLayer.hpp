#pragma once

#include <Layer/BaseLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuResizeLayer : public CuLayer, public ReSizeLayer {
    AOCE_LAYER_QUERYINTERFACE(CuResizeLayer)
   private:
    /* data */
    ImageType imageType = ImageType::rgba8;

   public:
    CuResizeLayer();
    CuResizeLayer(ImageType imageType);
    ~CuResizeLayer();

   protected:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace cuda
}  // namespace aoce
