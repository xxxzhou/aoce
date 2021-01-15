#include "FFmpegModule.hpp"

#include "AoceManager.hpp"
#include "FMediaPlayer.hpp"

namespace aoce {

FFmpegModule::FFmpegModule(/* args */) {}

FFmpegModule::~FFmpegModule() {}

bool FFmpegModule::loadModule() {
    auto version = avformat_version();
    if (version > 0) {
        AoceManager::Get().addMediaPlayerFactory(MediaPlayType::ffmpeg,
                                          new ffmpeg::FMediaPlayerFactory());
        return true;
    }
    return false;
}

void FFmpegModule::unloadModule() {
    AoceManager::Get().removeMediaPlayerFactory(MediaPlayType::ffmpeg);
}

ADD_MODULE(FFmpegModule, aoce_ffmpeg)
}  // namespace aoce