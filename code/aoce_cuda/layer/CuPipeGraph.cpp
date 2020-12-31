#include "CuPipeGraph.hpp"

namespace aoce {
namespace cuda {

CuPipeGraph::CuPipeGraph(/* args */) {
    AOCE_CUDEV_SAFE_CALL(cudaStreamCreate(&stream));
}

CuPipeGraph::~CuPipeGraph() {
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

cudaStream_t CuPipeGraph::getStream() { return stream; }

CudaMatRef CuPipeGraph::getOutTex(int32_t node, int32_t outIndex) {
    assert(node < nodes.size());
    CuLayer* cuLayer = static_cast<CuLayer*>(nodes[node]->getLayer());
    assert(outIndex < cuLayer->outCount);
    return cuLayer->outTexs[outIndex];
}

bool CuPipeGraph::onInitBuffers() { return true; }

bool CuPipeGraph::onRun() { return true; }

}  // namespace cuda
}  // namespace aoce