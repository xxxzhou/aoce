#pragma once
#include "Aoce.h"

namespace aoce {

enum class CaptureEventId : int32_t {
    other,
    lost,
    failed,
};

enum class WindowType : int32_t {
    other,
    win,
    android,
};

enum class CaptureType : int32_t {
    other,
    win_bitblt,
    win_rt,
};

class IWindow {
   public:
    IWindow(/* args */){};
    virtual ~IWindow(){};

    virtual const char* getTitle() = 0;
    virtual void* getHwnd() = 0;
    virtual uint64_t getProcessId() = 0;
    virtual uint64_t getMainThreadId() = 0;

    virtual bool bValid() = 0;
};

class ICaptureObserver {
   public:
    virtual ~ICaptureObserver(){};

    virtual void onEvent(CaptureEventId eventId, LogLevel level,
                         const char* msg){};
    virtual void onResize(int32_t width, int32_t height){};
    virtual void onCapture(const VideoFormat& videoFormat, void* device,
                           void* texture){};
};

class ICaptureWindow {
   public:
    virtual ~ICaptureWindow(){};

    virtual void setObserver(ICaptureObserver* observer) = 0;
    virtual bool bCapturing() = 0;
    // 初始化d3d device等信息
    virtual bool startCapture(IWindow* window, bool bSync) = 0;
    virtual bool renderCapture() = 0;
    virtual void stopCapture() = 0;
};

class IWindowManager {
   public:
    virtual ~IWindowManager(){};

    virtual int32_t getWindowCount(bool bUpdate = false) = 0;
    virtual IWindow* getWindow(int32_t index) = 0;
    virtual IWindow* getDesktop() = 0;
    virtual void setForeground(IWindow* window) = 0;
};

}  // namespace aoce