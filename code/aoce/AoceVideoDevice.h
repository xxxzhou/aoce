#pragma once
#include "Aoce.h"

namespace aoce {

#define AOCE_VIDEO_MAX_NAME 512

enum class VideoHandleId : int32_t {
    none = 0,
    open = 1,
    close = 2,
    unKnownError = -1,
    lost = -2,
    openFailed = -3,
};

class IVideoDeviceObserver {
   public:
    virtual ~IVideoDeviceObserver(){};
    virtual void onDeviceHandle(VideoHandleId id, int32_t codeId){};
    virtual void onVideoFrame(VideoFrame frame){};
    virtual void onDepthVideoFrame(VideoFrame colorFrame, VideoFrame depthFrame,
                                   void* alignParamt){};
};

class IVideoDevice {
   public:
    virtual ~IVideoDevice(){};
    virtual void setObserver(IVideoDeviceObserver* observer) = 0;

    // 得到支持格式的长度
    virtual int32_t getFormatCount() = 0;
    // 得到所有支持的格式
    virtual void getFormats(VideoFormat* formats, int32_t size,
                            int32_t start = 0) = 0;

    virtual const char* getName() = 0;
    virtual const char* getId() = 0;
    virtual const VideoFormat& getSelectFormat() = 0;

    // 针对android是否后置摄像头
    virtual bool back() = 0;
    // 是否深度摄像头
    virtual bool bDepth() = 0;

    // 选择一个最优解
    virtual int32_t findFormatIndex(int32_t width, int32_t height,
                                    int32_t fps = 30) = 0;

    // 摄像机有自己特定输出格式
    virtual void setFormat(int32_t index) = 0;
    // 摄像机没有特定输出格式,需要自己定义
    virtual void setFormat(const VideoFormat& format) = 0;
    // 打开摄像头
    virtual bool open() = 0;
    // 关闭摄像头
    virtual bool close() = 0;
    // 是否打开中
    virtual bool bOpen() = 0;
};

class IVideoManager {
   public:
    virtual ~IVideoManager(){};
    // 得到摄像头的个数
    virtual int32_t getDeviceCount(bool bUpdate = false) = 0;
    // 得到所有摄像头
    virtual void getDevices(IVideoDevice** videos, int32_t size,
                            int32_t start = 0) = 0;
};


}  // namespace aoce