#include "CuRGBA2YUVLayer.hpp"

#include "CuPipeGraph.hpp"

namespace aoce {
namespace cuda {

void rgb2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest,
                 int32_t yuvtype, cudaStream_t stream);
void rgb2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool ufront,
                 bool yfront, cudaStream_t stream);
void rgba2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest,
                  cudaStream_t stream);

CuRGBA2YUVLayer::CuRGBA2YUVLayer(/* args */) {}

CuRGBA2YUVLayer::~CuRGBA2YUVLayer() {}

void CuRGBA2YUVLayer::onInitLayer() {
    // 带P/SP的格式由r8转rgba8
    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
    if (paramet.type == VideoType::nv12 || paramet.type == VideoType::yuv420P) {
        outFormats[0].height = inFormats[0].height * 3 / 2;
    } else if (paramet.type == VideoType::yuy2P) {
        outFormats[0].height = inFormats[0].height * 2;
    } else if (paramet.type == VideoType::yuv2I ||
               paramet.type == VideoType::yvyuI ||
               paramet.type == VideoType::uyvyI) {
        outFormats[0].imageType = ImageType::rgba8;
        // 一个线程处理二个点,yuyv四点组合成一个元素,和rgba类似
        outFormats[0].width = inFormats[0].width / 2;
    }
}

bool CuRGBA2YUVLayer::onFrame() {
    if (paramet.type == VideoType::nv12 || paramet.type == VideoType::yuv420P ||
        paramet.type == VideoType::yuy2P) {
        int32_t yuvType = getYuvIndex(paramet.type);
        if (paramet.type == VideoType::yuy2P && paramet.special != 0) {
            rgba2yuv_gpu(*inTexs[0], *outTexs[0], stream);
        } else {
            rgb2yuv_gpu(*inTexs[0], *outTexs[0], yuvType, stream);
        }
    } else if (paramet.type == VideoType::yuv2I ||
               paramet.type == VideoType::yvyuI ||
               paramet.type == VideoType::uyvyI) {
        bool ufront = true;
        bool yfront = true;
        if (paramet.type == VideoType::yvyuI) {
            ufront = false;
        }
        if (paramet.type == VideoType::uyvyI) {
            yfront = false;
        }
        rgb2yuv_gpu(*inTexs[0], *outTexs[0], ufront, yfront, stream);
    }
    return true;
}

void CuRGBA2YUVLayer::onUpdateParamet() {
    assert(getYuvIndex(paramet.type) >= 0);
    if (pipeGraph) {
        pipeGraph->reset();
    }
}

}  // namespace cuda
}  // namespace aoce