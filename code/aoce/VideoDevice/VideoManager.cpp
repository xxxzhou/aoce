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

}  // namespace aoce