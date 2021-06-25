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

void InputLayer::checkImageFormat(const ImageFormat& imageFormat) {
    checkImageFormat(imageFormat, imageType2VideoType(imageFormat.imageType));
}

void InputLayer::checkImageFormat(const ImageFormat& newFormat,
                                  const VideoType& newVideo) {
    if (imageFormat == newFormat) {
        return;
    }
    assert(getLayer() != nullptr);
    assert(getLayer()->bAttachGraph());
    imageFormat = newFormat;
    videoType = newVideo;
    BaseLayer* layer = static_cast<BaseLayer*>(getLayer());
    layer->inFormats[0] = imageFormat;
    layer->outFormats[0] = imageFormat;
    // 输入层负责把bgra转化成rgba
    if (layer->outFormats[0].imageType == ImageType::bgra8) {
        layer->outFormats[0].imageType = ImageType::rgba8;
    }
    dataSize = imageFormat.width * imageFormat.height *
               getImageTypeSize(imageFormat.imageType);
    // 重新组织图
    layer->resetGraph();
}

void InputLayer::setImage(const ImageFormat& newFormat) {
    checkImageFormat(newFormat);
}

void InputLayer::setImage(const VideoFormat& newFormat) {
    ImageFormat iformat = videoFormat2ImageFormat(newFormat);
    checkImageFormat(iformat, newFormat.videoType);
}

void InputLayer::inputCpuData(uint8_t* data, bool bSeparateRun) {
    dataReady(data, bSeparateRun);
}

void InputLayer::inputCpuData(const VideoFrame& videoFrame, bool bSeparateRun) {
    VideoFormat vf = {};
    vf.width = videoFrame.width;
    vf.height = videoFrame.height;
    vf.videoType = videoFrame.videoType;
    ImageFormat iformat = videoFormat2ImageFormat(vf);
    checkImageFormat(iformat, vf.videoType);
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
    checkImageFormat(imageFormat);
    dataReady(data, bSeparateRun);
}

void InputLayer::inputGpuData(void* device, void* tex) {}

}  // namespace aoce