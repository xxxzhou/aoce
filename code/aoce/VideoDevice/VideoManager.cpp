#include "VideoManager.hpp"

namespace aoce {
VideoManager::VideoManager(/* args */) {}

VideoManager::~VideoManager() {}

const std::vector<VideoDevicePtr>& VideoManager::getDeviceList(bool bUpdate) {
    if (!oldUpdate || bUpdate) {
        getDevices();
        oldUpdate = true;
    }
    return videoList;
}

int32_t VideoManager::getDeviceCount(bool bUpdate) {
    if (!oldUpdate || bUpdate) {
        getDevices();
        oldUpdate = true;
    }
    return videoList.size();
}
// 得到所有支持的格式
void VideoManager::getDevices(IVideoDevice** videos, int32_t size,
                              int32_t start) {
    assert(start >= 0 && start < videoList.size());
    assert(start + size <= videoList.size());
    for (int32_t i = 0; i < size; i++) {
        *(videos + i) = videoList[i + start].get();
    }
}

IVideoDevice* VideoManager::getDevice(int32_t index) {
    assert(index >= 0 && index < videoList.size());
    return videoList[index].get();
}

}  // namespace aoce