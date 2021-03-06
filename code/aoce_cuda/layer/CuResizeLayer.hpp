#pragma once

#include <layer/BaseLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuResizeLayer : public CuLayer, public IReSizeLayer {
    AOCE_LAYER_QUERYINTERFACE(CuResizeLayer)
   private:
    /* data */
    ImageType imageType = ImageType::rgba8;

   public:
    CuResizeLayer();
    CuResizeLayer(ImageType imageType);
    virtual ~CuResizeLayer();

   protected:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace cuda
}  // namespace aoce
