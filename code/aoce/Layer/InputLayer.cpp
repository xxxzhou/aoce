#include "InputLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {

void InputLayer::setImage(VideoFormat videoFormat, int32_t index) {
    assert(this->getLayer() != nullptr);
    assert(this->getLayer()->getGraph() != nullptr);
    this->videoFormat = videoFormat;
    onSetImage(videoFormat, index);
    // 重新组织图
    this->getLayer()->getGraph()->reset();
}

void InputLayer::inputCpuData(uint8_t* data, int32_t index) {
    frameData = data;
    onInputCpuData(data, index);
}

void InputLayer::inputCpuData(const VideoFrame& videoFrame, int32_t index) {
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
    onInputCpuData(videoFrame, index);
}

}  // namespace aoce