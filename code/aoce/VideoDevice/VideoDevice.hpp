#pragma once
#include <functional>
#include <string>

#include "../Aoce.hpp"
// #include <Aoce.hpp>
#include <vector>

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

typedef std::function<void(VideoHandleId id, int32_t codeId)> deviceHandle;
typedef std::function<void(VideoFrame frame)> videoFrameHandle;
typedef std::function<void(VideoFrame colorFrame, VideoFrame depthFrame,
                           void* alignParamt)>
    depthFrameHandle;
typedef std::shared_ptr<class VideoDevice> VideoDevicePtr;

class ACOE_EXPORT VideoDevice {
   protected:
    /* data */
    int32_t selectIndex = 0;
    // 还不清楚编码,交给外面UI部分处理
    std::vector<char> name;
    std::vector<char> id;

    deviceHandle onDeviceEvent = nullptr;
    videoFrameHandle onVideoFrameEvent = nullptr;
    depthFrameHandle onDepthFrameEvent = nullptr;

    std::vector<VideoFormat> formats;
    VideoFormat selectFormat = {};
    bool isOpen = false;
    // 是否后置摄像头
    bool isBack = false;
    // 是否包含深度摄像头
    bool isDepth = false;

   public:
    VideoDevice(/* args */);
    virtual ~VideoDevice();

   protected:
    void onVideoFrameAction(VideoFrame frame);
    void onDepthFrameAction(VideoFrame colorFrame, VideoFrame depthFrame,
                            void* alignParamt);
    void onDeviceAction(VideoHandleId id, int32_t codeId);

   public:
    // 支持是否自己设置长宽
    bool bSetFormat() { return false; }
    const std::vector<VideoFormat>& getFormats() { return formats; };

    const std::vector<char>& getName() { return name; };
    const std::vector<char>& getId() { return id; };
    const VideoFormat& getSelectFormat() { return selectFormat; }

    bool back() { return isBack; }
    bool bDepth() { return isDepth; }

    // 选择一个最优解
    int32_t findFormatIndex(int32_t width, int32_t height, int32_t fps = 30);
    // 选择第一个满足width/height/filter的索引,否则为-1
    int32_t findFormatIndex(int32_t width, int32_t height,std::function<bool(VideoFormat)> filter);

   public:
    virtual void setVideoFrameHandle(videoFrameHandle handle);
    // 如果是深度摄像头
    virtual void setDepthFrameHandle(depthFrameHandle handle);
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