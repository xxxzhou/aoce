#pragma once
#include <layer/InputLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuMapChannelLayer : public CuLayer, public IMapChannelLayer {
    AOCE_LAYER_QUERYINTERFACE(CuMapChannelLayer)
   public:
    CuMapChannelLayer(){};
    virtual ~CuMapChannelLayer(){};

   protected:
    virtual bool onFrame() override;
};

class CuFlipLayer : public CuLayer, public IFlipLayer {
    AOCE_LAYER_QUERYINTERFACE(CuFlipLayer)
   public:
    CuFlipLayer(){};
    virtual ~CuFlipLayer(){};

   protected:
    virtual bool onFrame() override;
};

class CuTransposeLayer : public CuLayer, public ITransposeLayer {
    AOCE_LAYER_QUERYINTERFACE(CuTransposeLayer)
   public:
    CuTransposeLayer(){};
    virtual ~CuTransposeLayer(){};

   protected:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;
};

class CuBlendLayer : public CuLayer, public IBlendLayer {
    AOCE_LAYER_QUERYINTERFACE(CuBlendLayer)
   public:
    CuBlendLayer();
    virtual ~CuBlendLayer(){};

   private:
    std::unique_ptr<CudaMat> tempMat = nullptr;

   protected:
    virtual void onUpdateParamet() override;
    virtual bool onFrame() override;
    virtual void onInitCuBufffer() override;
};

}  // namespace cuda
}  // namespace aoce