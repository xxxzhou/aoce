#pragma once
#include <Layer/OutputLayer.hpp>

#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

// 现在CUDA与DX11交互,还是使用了DX11的共享纹理,需要一个独立的DX11上下文,后期考虑能不能替换掉,现在的问题是如果不使用
// 这种方式,渲染线程要等CUDA的流执行完,除非后期找到一种有锁的CUDA与Dx11线程交互方式才能保证各不等待并能正确执行.
class CuOutputLayer : public OutputLayer, public CuLayer {
    AOCE_LAYER_QUERYINTERFACE(CuOutputLayer)
   private:
    /* data */
    std::vector<uint8_t> cpuData;
    std::shared_ptr<win::Dx11SharedTex> shardTex;
    Dx11CudaResource cudaResoure = {};
    // CComPtr<ID3D11Device> device = nullptr;
    // CComPtr<ID3D11DeviceContext> ctx = nullptr;

   public:
    CuOutputLayer(/* args */);
    ~CuOutputLayer();

   public:
    virtual void onInitLayer() override;
    virtual bool onFrame() override;

   public:
    virtual void onInitCuBufffer() override;

   public:
    virtual void onUpdateParamet() override;
    virtual void outDx11GpuTex(void* device, void* tex) override;
};

}  // namespace cuda
}  // namespace aoce