#include "AVideoManager.hpp"
#include "AVideoDevice.hpp"
namespace aoce {
namespace android {

AVideoManager::AVideoManager(/* args */) {
    cameraManager = ACameraManager_create();
}

AVideoManager::~AVideoManager() {
    if (cameraIdList != nullptr) {
        ACameraManager_deleteCameraIdList(cameraIdList);
        cameraIdList = nullptr;
    }
}

void AVideoManager::getDevices() {
    videoList.clear();

    ACameraManager_getCameraIdList(cameraManager, &cameraIdList);
    for (int32_t index = 0; index < cameraIdList->numCameras; index++) {
        std::shared_ptr<AVideoDevice> videoPtr(new AVideoDevice());
        videoPtr->init(cameraManager,cameraIdList->cameraIds[index]);
        this->videoList.push_back(videoPtr);
    }
}

}  // namespace android
}  // namespace aoce