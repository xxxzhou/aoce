#include "CuInputLayer.hpp"

namespace aoce {
namespace cuda {

extern void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest,
                         cudaStream_t stream);
extern void argb2rgba_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                          cudaStream_t stream);

CuInputLayer::CuInputLayer(/* args */) {}

CuInputLayer::~CuInputLayer() {}

void CuInputLayer::onSetImage(VideoFormat videoFormat, int32_t index) {
    assert(index < inCount);
    // 根据各种格式调整(比如YUV格式,长宽要变成YUV本身的长度)
    inFormats[index] = videoFormat2ImageFormat(videoFormat);
}

void CuInputLayer::onInitCuBufffer() {
    if (videoFormat.videoType == VideoType::rgb8) {
        tempMat = std::shared_ptr<CudaMat>(new CudaMat());
        tempMat->create(inFormats[0].width, inFormats[0].height, AOCE_CV_8UC3);
    } else if (videoFormat.videoType == VideoType::bgra8) {
        tempMat = std::shared_ptr<CudaMat>(new CudaMat());
        tempMat->create(inFormats[0].width, inFormats[0].height, AOCE_CV_8UC4);
    }
}

bool CuInputLayer::onFrame() {
    if (this->videoFormat.videoType == VideoType::rgb8) {
        tempMat->upload(frameData, 0, stream);
        rgb2rgba_gpu(tempMat.get(), outTexs[0].get(), stream);

    } else if (videoFormat.videoType == VideoType::bgra8) {
        tempMat->upload(frameData, 0, stream);
    }

    return true;
}

}  // namespace cuda
}  // namespace aoce