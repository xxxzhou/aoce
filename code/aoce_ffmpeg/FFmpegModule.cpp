#include "FFmpegModule.hpp"

#include "AoceFFmpeg.hpp"
#include "AoceManager.hpp"
#include "FFmpegModule.hpp"
#include "FMediaFactory.hpp"

namespace aoce {

FFmpegModule::FFmpegModule(/* args */) {}

FFmpegModule::~FFmpegModule() {}

bool FFmpegModule::loadModule() {
    auto version = avformat_version();
    uint32_t major = AV_VERSION_MAJOR(version);
    uint32_t minor = AV_VERSION_MINOR(version);
    uint32_t micro = AV_VERSION_MICRO(version);
    if (version > 0) {
        std::string msg = "";
        string_format(msg, "avformat version:", major, ".", minor, ".", micro);
        logMessage(LogLevel::info, msg);
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