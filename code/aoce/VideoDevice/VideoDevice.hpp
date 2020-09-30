#pragma once
#include <functional>
#include <string>

#include "../Aoce.hpp"
#include "vector"

namespace aoce {

enum class VideoHandleId : int32_t {
    none,
    open,
    close,
    unKnownError = -1,
    lost = -2,
    openFailed = -3,
};

typedef std::function<void(VideoHandleId id, int32_t codeId)> deviceHandle;
typedef std::function<void(VideoFrame frame)> videoFrameHandle;

class ACOE_EXPORT VideoDevice {
   protected:
    /* data */
    int32_t selectIndex = 0;
    // 还不清楚编码,交给外面UI部分处理
    std::vector<char> name;
    std::vector<char> id;

    deviceHandle onDeviceEvent = nullptr;
    videoFrameHandle onVideoFrameEvent = nullptr;

    std::vector<VideoFormat> formats;
    VideoFormat selectFormat;
    bool isOpen = false;

   public:
    VideoDevice(/* args */);
    virtual ~VideoDevice();

   protected:
    void onVideoFrameAction(VideoFrame frame);
    void onDeviceAction(VideoHandleId id, int32_t codeId);

   public:
    // 支持是否自己设置长宽
    bool bSetFormat() { return false; }
    const std::vector<VideoFormat>& getFormats() { return formats; };

    const std::vector<char>& getName() { return name; };
    const std::vector<char>& getId() { return id; };
    const VideoFormat& getSelectFormat() { return selectFormat; }

   public:
    virtual void setVideoFrameHandle(videoFrameHandle handle);
    virtual void setDeviceHandle(deviceHandle handle);

    // 摄像机有自己特定输出格式
    virtual void setFormat(int32_t index){};
    // 摄像机没有特定输出格式,需要自己定义
    virtual void setFormat(const VideoFormat& format){};
    // 打开摄像头
    virtual bool open() { return false; };
    // 关闭摄像头
    virtual bool close() { return false; };
    // 是否打开中
    virtual bool bOpen() { return isOpen; }
};

}  // namespace aoce