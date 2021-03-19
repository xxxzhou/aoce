#include "CuResizeLayer.hpp"

namespace aoce {
namespace cuda {

template <typename T>
void resize_gpu(PtrStepSz<T> source, PtrStepSz<T> dest, bool bLinear,
                cudaStream_t stream);

CuResizeLayer::CuResizeLayer(/* args */) : CuResizeLayer(ImageType::rgba8) {}

CuResizeLayer::CuResizeLayer(ImageType imageType) {
    this->imageType = imageType;
    paramet.bLinear = true;
    paramet.newWidth = 1920;
    paramet.newHeight = 1080;
}

CuResizeLayer::~CuResizeLayer() {}

void CuResizeLayer::onInitLayer() {
    assert(paramet.newWidth > 0 && paramet.newHeight > 0);
    outFormats[0].width = paramet.newWidth;
    outFormats[0].height = paramet.newHeight;
}

bool CuResizeLayer::onFrame() {
    if (imageType == ImageType::r8) {
        resize_gpu<uchar>(*inTexs[0], *outTexs[0], paramet.bLinear, stream);
    } else if (imageType == ImageType::rgba8) {
        resize_gpu<uchar4>(*inTexs[0], *outTexs[0], paramet.bLinear, stream);
    } else if (imageType == ImageType::rgbaf32) {
        resize_gpu<float4>(*inTexs[0], *outTexs[0], paramet.bLinear, stream);
    }
    return true;
}

void CuResizeLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    resetGraph();
}

}  // namespace cuda
}  // namespace aoce