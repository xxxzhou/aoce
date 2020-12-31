#pragma once
#include <Layer/InputLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuInputLayer : public InputLayer, public CuLayer {
    AOCE_LAYER_QUERYINTERFACE(CuInputLayer)
   private:
    /* data */
    CudaMatRef tempMat = nullptr;

   public:
    CuInputLayer(/* args */);
    ~CuInputLayer();

    // InputLayer
   public:
    virtual void onSetImage(VideoFormat videoFormat,
                            int32_t index = 0) override;
    virtual void onInputCpuData(uint8_t* data, int32_t index = 0) override{};
    virtual void onInputCpuData(const VideoFrame& videoFrame,
                                int32_t index = 0) override{};

   public:
    virtual void onInitCuBufffer() override;
    virtual bool onFrame() override;
};

}  // namespace cuda
}  // namespace aoce