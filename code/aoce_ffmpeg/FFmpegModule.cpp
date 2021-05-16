#include "FFmpegModule.hpp"

#include "AoceManager.hpp"
#include "FMediaPlayer.hpp"

namespace aoce {

FFmpegModule::FFmpegModule(/* args */) {}

FFmpegModule::~FFmpegModule() {}

bool FFmpegModule::loadModule() {
    auto version = avformat_version();
    if (version > 0) {
        AoceManager::Get().addMediaFactory(MediaType::ffmpeg,
                                          new ffmpeg::FMediaFactory());
        return true;
    }
    return false;
}

void FFmpegModule::unloadModule() {
    AoceManager::Get().removeMediaFactory(MediaType::ffmpeg);
}

ADD_MODULE(FFmpegModule, aoce_ffmpeg)
}  // namespace aoce