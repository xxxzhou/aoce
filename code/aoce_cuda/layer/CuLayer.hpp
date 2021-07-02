#pragma once
#include <layer/BaseLayer.hpp>

#include "../CudaHelper.hpp"
#include "../CudaMat.hpp"

namespace aoce {
namespace cuda {

class AOCE_CUDA_EXPORT CuLayer : public BaseLayer {
    friend class CuPipeGraph;

   private:
    /* data */

   protected:
    int32_t groupX = 32;
    int32_t groupY = 8;
    class CuPipeGraph* cuPipeGraph = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<CudaMatRef> inTexs;
    std::vector<CudaMatRef> outTexs;

   public:
    CuLayer(/* args */);
    ~CuLayer();

   protected:
    virtual void onInit() final;
    // virtual void onInitLayer() override;
    virtual void onInitBuffer() final;
    virtual bool onFrame() override;
    virtual void onUnInit() override;

   protected:
    virtual void onInitCuBufffer(){};
};

}  // namespace cuda
}  // namespace aoce