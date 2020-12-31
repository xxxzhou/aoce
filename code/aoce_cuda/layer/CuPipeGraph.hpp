#pragma once

#include <Layer/PipeGraph.hpp>

#include "../CudaMat.hpp"
#include "CuLayer.hpp"

namespace aoce {
namespace cuda {

class AOCE_CUDA_EXPORT CuPipeGraph : public PipeGraph {
   private:
    /* data */
    cudaStream_t stream = nullptr;
    std::vector<CuLayer*> cuLayers;

   public:
    CuPipeGraph(/* args */);
    ~CuPipeGraph();

   public:
    cudaStream_t getStream();
    CudaMatRef getOutTex(int32_t node, int32_t outIndex);

   protected:
    // 所有layer调用initbuffer后
    virtual bool onInitBuffers();
    virtual bool onRun();
};

class CuPipeGraphFactory : public PipeGraphFactory {
   public:
    CuPipeGraphFactory(){};
    virtual ~CuPipeGraphFactory(){};

   public:
    inline virtual PipeGraph* createGraph() override {
        return new CuPipeGraph();
    };
};

}  // namespace cuda
}  // namespace aoce
