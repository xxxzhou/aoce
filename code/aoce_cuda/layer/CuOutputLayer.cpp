#include "CuOutputLayer.hpp"

namespace aoce {
namespace cuda {

CuOutputLayer::CuOutputLayer(/* args */) { bOutput = true; }

CuOutputLayer::~CuOutputLayer() {}

void CuOutputLayer::onInitLayer() {
    if (paramet.bCpu) {
        int32_t imageSize = getImageTypeSize(inFormats[0].imageType);
        cpuData.resize(inFormats[0].width * inFormats[0].height * imageSize);
    }
}

bool CuOutputLayer::onFrame() {
    inTexs[0]->download(cpuData.data(), 0, stream);
    cudaStreamSynchronize(stream);
    if (paramet.bCpu) {
        onImageProcessHandle(cpuData.data(), inFormats[0].width,
                             inFormats[0].height, 0);
    }
    return true;
}

}  // namespace cuda
}  // namespace aoce
