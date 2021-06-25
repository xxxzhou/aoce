#pragma once
#include <functional>
#include <string>

#include "../Aoce.hpp"
// #include <Aoce.hpp>
#include <vector>

namespace aoce {

class ACOE_EXPORT AudioDevice : public IVideoDevice {
   private:
    /* data */
    std::string name = "";
    std::string id = "";
    AudioFormat audioFormat = {};

    IAudioDeviceObserver* observer = nullptr;
    AudioDeviceType deviceType = AudioDeviceType::other;

   public:
    AudioDevice(/* args */);
    virtual ~AudioDevice();

   public:
    virtual AudioDeviceType getDeviceType() final;
    virtual const char* getName() final;
    virtual const char* getId() final;

    virtual void setObserver(IAudioDeviceObserver* observer) final;
    // 初始化后得到使用格式
    virtual const AudioFormat& getAudioFormat() final;
};

}  // namespace aoce