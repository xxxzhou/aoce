#pragma once

#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCameraDevice.h>
#include <media/NdkImageReader.h>
#include "VideoDevice/VideoManager.hpp"

namespace aoce {
namespace android {

class AVideoManager : public VideoManager {
   private:
    /* data */
    ACameraManager* cameraManager = nullptr;
    ACameraIdList* cameraIdList = nullptr;
   public:
    AVideoManager(/* args */);
    virtual ~AVideoManager() override;

   protected:
    virtual void getDevices() override;
};

}  // namespace android
}  // namespace aoce