#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "aoce/AoceCore.h"

namespace aoce {

class CameraManager {
   private:
    /* data */
    CameraManager(/* args */);
    static CameraManager *instance;

   private:
    std::vector<IVideoDevice *> deviceList;

   public:
    static CameraManager &Get();
    ~CameraManager();

   public:
    std::vector<IVideoDevice *> getVideoList();
    IVideoDevice *getVideoDevice(const char *deviceId);
};

}  // namespace aoce
