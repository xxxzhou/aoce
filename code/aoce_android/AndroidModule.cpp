#include "AndroidModule.hpp"

#include <AoceManager.hpp>

#include "AVideoManager.hpp"

namespace aoce {
namespace android {

AndroidModule::AndroidModule(/* args */) {}

AndroidModule::~AndroidModule() {}

bool AndroidModule::loadModule() {
    AoceManager::Get().addVideoManager(CameraType::and_camera2,
                                       new AVideoManager());
    return true;
}

void AndroidModule::unloadModule() {
    AoceManager::Get().removeVideoManager(CameraType::and_camera2);
}

ADD_MODULE(AndroidModule, aoce_android)

}  // namespace android
}  // namespace aoce