#include "VideoDevice.hpp"

namespace aoce {
VideoDevice::VideoDevice(/* args */) {
    name.resize(512, 0);
    id.resize(512, 0);
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

}  // namespace aoce