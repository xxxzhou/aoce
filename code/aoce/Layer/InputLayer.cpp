#include "InputLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {

void InputLayer::checkImageFormat(int32_t width, int32_t height,
                                  VideoType videoType) {
    assert(getLayer() != nullptr);
    assert(getLayer()->getGraph() != nullptr);
    assert(getLayer()->inFormats.size() > 0);
    if (width != videoFormat.width || height != videoFormat.height ||
        videoType != videoFormat.videoType) {
        videoFormat.width = width;
        videoFormat.height = height;
        videoFormat.videoType = videoType;
        getLayer()->inFormats[0] = videoFormat2ImageFormat(videoFormat);
        getLayer()->outFormats[0] = getLayer()->inFormats[0];
        // 重新组织图
        getLayer()->resetGraph();
    }
}

void InputLayer::setImage(VideoFormat newFormat) {
    checkImageFormat(newFormat.width, newFormat.height, newFormat.videoType);
}

void InputLayer::inputCpuData(uint8_t* data) {
    frameData = data;
    onDataReady();
}

void InputLayer::inputCpuData(const VideoFrame& videoFrame) {
    checkImageFormat(videoFrame.width, videoFrame.height, videoFrame.videoType);   
    int32_t size = getVideoFrame(videoFrame, nullptr);
    if (size == 0) {
        frameData = videoFrame.data[0];
    } else {
        if (videoFrameData.size() != size) {
            videoFrameData.resize(size);
        }
        getVideoFrame(videoFrame, videoFrameData.data());
        frameData = videoFrameData.data();
    }
    onDataReady();
}

void InputLayer::inputCpuData(uint8_t* data, const ImageFormat& imageFormat) {    
    VideoType vty = imageType2VideoType(imageFormat.imageType);
    checkImageFormat(imageFormat.width, imageFormat.height, vty);
    frameData = data;
    onDataReady();
}

}  // namespace aoce