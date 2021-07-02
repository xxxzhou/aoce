#include "CuLayer.hpp"

#include "../CudaHelper.hpp"
#include "CuPipeGraph.hpp"

namespace aoce {
namespace cuda {

CuLayer::CuLayer(/* args */) {
    gpu = GpuType::cuda;
}

CuLayer::~CuLayer() {}

void CuLayer::onInit() {
    BaseLayer::onInit();
    cuPipeGraph = static_cast<CuPipeGraph*>(pipeGraph);
    stream = cuPipeGraph->getStream();
}

void CuLayer::onInitBuffer() {
    // 得到上层的结果
    if (!bInput) {
        inTexs.clear();
        for (int32_t i = 0; i < inCount; i++) {
            auto& inLayer = this->inLayers[i];
            inTexs.push_back(
                cuPipeGraph->getOutTex(inLayer.nodeIndex, inLayer.siteIndex));
        }
    }
    // 当前层计算结果
    if (!bOutput) {
        outTexs.clear();
        for (int32_t i = 0; i < outCount; i++) {
            const ImageFormat& format = outFormats[i];
            int32_t flag = ImageFormat2Cuda(format.imageType);
            CudaMatRef texPtr = std::shared_ptr<CudaMat>(new CudaMat());
            texPtr->create(format.width, format.height, flag);
            outTexs.push_back(texPtr);
        }
    }
    onInitCuBufffer();
}

bool CuLayer::onFrame() { return true; }

void CuLayer::onUnInit() {
    for (int i = 0; i < outCount; i++) {
        outTexs[i].reset();
    }
    outTexs.clear();
}

}  // namespace cuda
}  // namespace aoce