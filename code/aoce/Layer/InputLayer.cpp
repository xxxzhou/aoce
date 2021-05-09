#include "InputLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {

void InputLayer::dataReady(uint8_t* data, bool bCopy) {
    if (bCopy) {
        ImageFormat inFormat = getLayer()->inFormats[0];
        int32_t size = inFormat.width * inFormat.height *
                       getImageTypeSize(inFormat.imageType);
        if (size <= 0) {
            std::string message;
            string_format(message, getLayer()->getMark(),
                          " in format incorrect");
            logMessage(LogLevel::error, message.c_str());
            assert(false);
        }
        if (videoFrameData.size() != size) {
            videoFrameData.resize(size);
        }
        memcpy(videoFrameData.data(), data, size);
        frameData = videoFrameData.data();
    } else {
        frameData = data;
    }
    onDataReady();
}

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

void InputLayer::inputCpuData(uint8_t* data, bool bSeparateRun) {
    dataReady(data, bSeparateRun);
}

void InputLayer::inputCpuData(const VideoFrame& videoFrame, bool bSeparateRun) {
    checkImageFormat(videoFrame.width, videoFrame.height, videoFrame.videoType);
    int32_t size = getVideoFrame(videoFrame, nullptr);
    if (size == 0) {
        dataReady(videoFrame.data[0], bSeparateRun);
    } else {
        if (videoFrameData.size() != size) {
            videoFrameData.resize(size);
        }
        getVideoFrame(videoFrame, videoFrameData.data());
        dataReady(videoFrameData.data(), false);
    }
}

void InputLayer::inputCpuData(uint8_t* data, const ImageFormat& imageFormat,
                              bool bSeparateRun) {
    VideoType vty = imageType2VideoType(imageFormat.imageType);
    checkImageFormat(imageFormat.width, imageFormat.height, vty);
    dataReady(data, bSeparateRun);
}

}  // namespace aoce