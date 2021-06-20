#include "CuPipeGraph.hpp"

using namespace aoce::win;

namespace aoce {
namespace cuda {

CuPipeGraph::CuPipeGraph(/* args */) {
    gpu = GpuType::cuda;
    AOCE_CUDEV_SAFE_CALL(cudaStreamCreate(&stream));
    createDevice11(&device, &ctx);
}

CuPipeGraph::~CuPipeGraph() {
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

ID3D11Device* CuPipeGraph::getDX11Device(){
    return device;
}

cudaStream_t CuPipeGraph::getStream() { return stream; }

CudaMatRef CuPipeGraph::getOutTex(int32_t node, int32_t outIndex) {
    assert(node < nodes.size());
    CuLayer* cuLayer = static_cast<CuLayer*>(nodes[node]->getLayer());
    assert(outIndex < cuLayer->outCount);
    return cuLayer->outTexs[outIndex];
}

void CuPipeGraph::onReset(){
    cuLayers.clear();
}

bool CuPipeGraph::onInitBuffers() {
    cuLayers.clear();
    for (auto index : nodeExcs) {
        CuLayer* cuLayer = static_cast<CuLayer*>(nodes[index]->getLayer());
        cuLayers.push_back(cuLayer);
    }
    return true;
}

bool CuPipeGraph::onRun() {
    for (auto* layer : cuLayers) {
        if (!layer->onFrame()) {
            return false;
        }
    }
    return true;
}

}  // namespace cuda
}  // namespace aoce