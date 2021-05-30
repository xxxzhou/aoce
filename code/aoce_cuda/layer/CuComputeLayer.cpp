#include "CuComputeLayer.hpp"

namespace aoce {
namespace cuda {

void textureMap_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                    MapChannelParamet paramt, cudaStream_t stream);
void blend_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> blendTex,
               PtrStepSz<uchar4> dest, int32_t left, int32_t top, float opacity,
               cudaStream_t stream);
void flip_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
              FlipParamet paramt, cudaStream_t stream);
void transpose_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                   TransposeParamet paramt, cudaStream_t stream);
void gamma_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, float gamma,
               cudaStream_t stream);
template <typename T>
void resize_gpu(PtrStepSz<T> source, PtrStepSz<T> dest, bool bLinear,
                cudaStream_t stream);

bool CuMapChannelLayer::onFrame() {
    textureMap_gpu(*inTexs[0], *outTexs[0], paramet, stream);
    return true;
}

bool CuFlipLayer::onFrame() {
    flip_gpu(*inTexs[0], *outTexs[0], paramet, stream);
    return true;
}

void CuTransposeLayer::onInitLayer() {
    outFormats[0].width = inFormats[0].height;
    outFormats[0].height = inFormats[0].width;
}

bool CuTransposeLayer::onFrame() {
    transpose_gpu(*inTexs[0], *outTexs[0], paramet, stream);
    return true;
}

CuBlendLayer::CuBlendLayer() { inCount = 2; }

void CuBlendLayer::onUpdateParamet() {
    if (paramet.width != oldParamet.width ||
        paramet.height != oldParamet.height) {
        resetGraph();
    }
}

bool CuBlendLayer::onFrame() {
    float top = (paramet.centerY - paramet.height / 2) * inFormats[0].width;
    float left = (paramet.centerX - paramet.width / 2) * inFormats[0].height;
    resize_gpu<uchar4>(*inTexs[0], *tempMat, true, stream);
    blend_gpu(*inTexs[0], *tempMat, *outTexs[0], left, top, paramet.alaph,
              stream);
    return true;
}

void CuBlendLayer::onInitCuBufffer() {
    int32_t tempWidth = paramet.width * inFormats[0].width;
    int32_t tempHeight = paramet.height * inFormats[0].height;
    tempMat = std::make_unique<CudaMat>();
    tempMat->create(tempWidth, tempHeight, AOCE_CV_8UC4);
}

}  // namespace cuda
}  // namespace aoce