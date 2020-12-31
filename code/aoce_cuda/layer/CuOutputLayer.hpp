#pragma once
#include <Layer/OutputLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuOutputLayer : public OutputLayer, public CuLayer {
    AOCE_LAYER_QUERYINTERFACE(CuOutputLayer)
   private:
    /* data */
    std::vector<uint8_t> cpuData;

   public:
    CuOutputLayer(/* args */);
    ~CuOutputLayer();

   public:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;
};

}  // namespace cuda
}  // namespace aoce