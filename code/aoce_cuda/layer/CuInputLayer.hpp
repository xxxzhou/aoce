#pragma once
#include <layer/InputLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class CuInputLayer : public InputLayer, public CuLayer {
    AOCE_LAYER_QUERYINTERFACE(CuInputLayer)
   private:
    /* data */
    CudaMatRef tempMat = nullptr;

    std::shared_ptr<win::Dx11SharedTex> shardTex;
    Dx11CudaResource cudaResoure = {};
    // CComPtr<ID3D11Device> device = nullptr;
    // CComPtr<ID3D11DeviceContext> ctx = nullptr;

   public:
    CuInputLayer(/* args */);
    ~CuInputLayer();

    // InputLayer
   public:
    virtual void onDataReady() override{};
    virtual void inputGpuData(void* device, void* tex) override;

   public:
    virtual void onUpdateParamet() override;

   protected:
    virtual void onInitCuBufffer() override;
    virtual bool onFrame() override;
    virtual void onUnInit() override;
};

}  // namespace cuda
}  // namespace aoce