#include "VideoDevice.hpp"

namespace aoce {
VideoDevice::VideoDevice(/* args */) {
    name.resize(AOCE_VIDEO_MAX_NAME, 0);
    id.resize(AOCE_VIDEO_MAX_NAME, 0);
}

VideoDevice::~VideoDevice() {}

void VideoDevice::onVideoFrameAction(VideoFrame frame) {
    if (onVideoFrameEvent) {
        onVideoFrameEvent(frame);
    }
}

void VideoDevice::onDeviceAction(VideoHandleId id, int32_t codeId) {
    if (onDeviceEvent) {
        onDeviceEvent(id, codeId);
    }
}

void VideoDevice::setVideoFrameHandle(videoFrameHandle handle) {
    onVideoFrameEvent = handle;
}

void VideoDevice::setDeviceHandle(deviceHandle handle) {
    onDeviceEvent = handle;
}

int32_t VideoDevice::findFormatIndex(int32_t width, int32_t height,
                                     int32_t fps) {
    int32_t index = 0;
    if (formats.size() < 0) {
        return -1;
    }
    bool bFind = false;
    int32_t first = -1;
    int32_t second = -1;
    for (const VideoFormat& format : formats) {
        if (format.width == width && format.height == height) {
            bFind = true;
            // 如果全满足,直接返回
            if (format.fps == fps && format.videoType != VideoType::mjpg) {
                return index;
            }
            // 尽量不选MJPG,多了解码的消耗
            if (format.videoType != VideoType::mjpg) {
                if (first < 0 || formats[first].fps > format.fps) {
                    first = index;
                }
            } else {
                if (second < 0 || formats[second].fps > format.fps) {
                    second = index;
                }
            }
        }
        index++;
    }
    if (bFind) {
        return first >= 0 ? first : second;
    }
    return 0;
}

}  // namespace aoce