#pragma once
#include "../Aoce.hpp"
#include "VideoDevice.hpp"
namespace aoce {

class ACOE_EXPORT VideoManager : public IVideoManager {
   protected:
    /* data */
    std::vector<VideoDevicePtr> videoList;
    bool oldUpdate = false;

   public:
    VideoManager(/* args */);
    virtual ~VideoManager();

   protected:
    virtual void getDevices(){};

   public:
    const std::vector<VideoDevicePtr>& getDeviceList(bool bUpdate = false);

   public:
    virtual int32_t getDeviceCount(bool bUpdate = false) override;
    // 得到所有支持的格式
    virtual void getDevices(IVideoDevice** videos, int32_t size,
                            int32_t start = 0) override;
    virtual IVideoDevice* getDevice(int32_t index) override;
};

}  // namespace aoce