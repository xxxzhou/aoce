#include "VideoManager.hpp"

namespace aoce {
VideoManager::VideoManager(/* args */) {}

VideoManager::~VideoManager() {}

const std::vector<VideoDevicePtr>& VideoManager::getDeviceList() {
    getDevices();
    return videoList;
}

}  // namespace aoce