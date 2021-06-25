#include "AudioDevice.hpp"

namespace aoce {

AudioDevice::AudioDevice(/* args */) {}

AudioDevice::~AudioDevice() {}

AudioDeviceType AudioDevice::getDeviceType() { return deviceType; }

const char* AudioDevice::getName() { return name.c_str(); }

const char* AudioDevice::getId() { return id.c_str(); }

void AudioDevice::setObserver(IAudioDeviceObserver* observer) {
    this->observer = observer;
}

const AudioFormat& AudioDevice::getAudioFormat() { return audioFormat; }

}  // namespace aoce