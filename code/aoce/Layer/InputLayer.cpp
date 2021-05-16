#include "InputLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {

void InputLayer::dataReady(uint8_t* data, bool bCopy) {
    if (bCopy) {
        if (dataSize <= 0) {
            std::string message;
            string_format(message, getLayer()->getMark(),
                          " in format incorrect");
            logMessage(LogLevel::error, message.c_str());
            assert(false);
        }
        if (videoFrameData.size() != dataSize) {
            videoFrameData.resize(dataSize);
        }
        memcpy(videoFrameData.data(), data, dataSize);
        frameData = videoFrameData.data();
    } else {
        frameData = data;
    }
    onDataReady();
}

void InputLayer::checkImageFormat(int32_t width, int32_t height,
                                  VideoType videoType) {
    assert(getLayer() != nullptr);
    assert(getLayer()->bAttachGraph());
    if (width != videoFormat.width || height != videoFormat.height ||
        videoType != videoFormat.videoType) {
        videoFormat.width = width;
        videoFormat.height = height;
        videoFormat.videoType = videoType;
        ImageFormat iformat = videoFormat2ImageFormat(videoFormat);
        BaseLayer* layer = static_cast<BaseLayer*>(getLayer());
        layer->inFormats[0] = iformat;
        layer->outFormats[0] = iformat;
        dataSize = iformat.width * iformat.height *
                   getImageTypeSize(iformat.imageType);
        // 重新组织图
        layer->resetGraph();
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