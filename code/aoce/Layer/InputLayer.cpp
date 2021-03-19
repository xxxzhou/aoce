#include "InputLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {

void InputLayer::setImage(VideoFormat videoFormat, int32_t index) {
    assert(this->getLayer() != nullptr);
    assert(this->getLayer()->getGraph() != nullptr);
    assert(this->getLayer()->inFormats.size() > 0);
    this->videoFormat = videoFormat;
    this->getLayer()->inFormats[0] = videoFormat2ImageFormat(videoFormat);
    this->getLayer()->outFormats[0] = this->getLayer()->inFormats[0];
    // 重新组织图
    this->getLayer()->resetGraph();
}

void InputLayer::inputCpuData(uint8_t* data, int32_t index) {
    frameData = data;
    onDataReady();
}

void InputLayer::inputCpuData(const VideoFrame& videoFrame, int32_t index) {
    if (videoFormat.width != videoFrame.width ||
        videoFormat.height != videoFrame.height ||
        videoFormat.videoType != videoFrame.videoType) {
        videoFormat.width = videoFrame.width;
        videoFormat.height = videoFrame.height;
        videoFormat.videoType = videoFrame.videoType;
        setImage(videoFormat, index);
    }
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
    BaseLayer* layer = getLayer();
    assert(layer != nullptr);
    assert(layer->inFormats.size() > 0);
    ImageFormat oldFormat = layer->inFormats[0];
    if (oldFormat.width != imageFormat.width ||
        oldFormat.height != imageFormat.height ||
        oldFormat.imageType != imageFormat.imageType) {
        layer->inFormats[0] = imageFormat;
        layer->outFormats[0] = imageFormat;
        layer->resetGraph();
    }
    frameData = data;
    onDataReady();
}

}  // namespace aoce