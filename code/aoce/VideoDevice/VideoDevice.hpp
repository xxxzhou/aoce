#pragma once
#include <functional>
#include <string>

#include "../Aoce.hpp"
// #include <Aoce.hpp>
#include <vector>

namespace aoce {

typedef std::shared_ptr<class VideoDevice> VideoDevicePtr;
typedef std::weak_ptr<class VideoDevice> VideoDeviceWeak;

class ACOE_EXPORT VideoDevice : public IVideoDevice {
   protected:
    /* data */
    int32_t selectIndex = 0;
    // 还不清楚编码,交给外面UI部分处理
    std::string name = "";
    std::string id = "";

    IVideoDeviceObserver* observer = nullptr;

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
    inline bool bSetFormat() { return false; }
    inline const std::vector<VideoFormat>& getFormats() { return formats; };

    inline virtual void setObserver(IVideoDeviceObserver* observer) final {
        this->observer = observer;
    };

    inline virtual int32_t getFormatCount() final { return formats.size(); }
    inline virtual void getFormats(VideoFormat* formatList, int32_t size,
                                   int32_t start = 0) final {
        assert(start >= 0 && start < formats.size());
        assert(start + size <= formats.size());
        memcpy(formatList, &formats[start], sizeof(VideoFormat) * size);
    }

    inline virtual const char* getName() final { return name.c_str(); };
    inline virtual const char* getId() final { return id.c_str(); };
    inline virtual const VideoFormat& getSelectFormat() final {
        return selectFormat;
    }

    inline virtual bool back() final { return isBack; }
    inline virtual bool bDepth() final { return isDepth; }

    // 选择一个最优解
    inline virtual int32_t findFormatIndex(int32_t width, int32_t height,
                                           int32_t fps = 30) final;
    // 选择第一个满足width/height/filter的索引,否则为-1
    int32_t findFormatIndex(int32_t width, int32_t height,
                            std::function<bool(VideoFormat)> filter);

    // 摄像机有自己特定输出格式
    virtual void setFormat(int32_t index) override {}
    // 摄像机没有特定输出格式,需要自己定义
    virtual void setFormat(const VideoFormat& format) override {}
    // 打开摄像头
    virtual bool open() override { return false; }
    // 关闭摄像头
    virtual bool close() override { return false; }
    // 是否打开中
    virtual bool bOpen() override { return isOpen; }
};

}  // namespace aoce