#pragma once
#include "Aoce.h"

namespace aoce {

enum class AudioHandleId : int32_t {
    none = 0,
    open = 1,
    close = 2,
    unKnownError = -1,
    lost = -2,
    openFailed = -3,
};

enum class AudioDeviceType : int32_t {
    other,
    // 麦克风
    mic,
    // 播放
    player,
    // 声卡输出
    loopback,
};

struct AudioFormat {
    int32_t channel = 1;
    // 8000,11025,22050,44100
    int32_t sampleRate = 8000;
    // 16，24，32
    int32_t depth = 16;
};

class IAudioDeviceObserver {
   public:
    IAudioDeviceObserver() = default;
    virtual ~IAudioDeviceObserver(){};

   public:
    virtual void onDeviceHandle(VideoHandleId id, int32_t codeId){};
    virtual void onAudioFrame(const AudioFrame& audioFrame){};
};

class IAudioDevice {
   public:
    // IAudioDevice(/* args */) = default;
    virtual ~IAudioDevice(){};

    virtual AudioDeviceType getDeviceType() = 0;
    virtual const char* getName() = 0;
    virtual const char* getId() = 0;

    virtual void setObserver(IAudioDeviceObserver* observer) = 0;
    // 设备相关资源初始化
    virtual void prepare(bool bAsync) = 0;
    // 初始化后得到使用格式
    virtual const AudioFormat& getAudioFormat() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void release() = 0;
};

}  // namespace aoce