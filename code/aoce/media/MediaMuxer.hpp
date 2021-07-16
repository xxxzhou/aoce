#pragma once

#include "../Aoce.hpp"
#include "../module/FrameCount.hpp"
#include "MediaHelper.hpp"

namespace aoce {

class ACOE_EXPORT MediaMuxer : public IMediaMuxer {
   protected:
    /* data */
    std::string url = "";
    AudioStream audioStream = {};
    VideoStream videoStream = {};
    bool bAudio = false;
    bool bVideo = false;
    MediaSourceType mediaType = MediaSourceType::other;
    //统计时间
    FrameCount videoFps = {};
    //统计码率
    FrameCount videoRate = {};

   public:
    MediaMuxer(/* args */);
    virtual ~MediaMuxer();

   public:
    virtual void setOutput(const char* path) final;
    virtual void setAudioStream(const AudioStream& stream) final;
    virtual void setVideoStream(const VideoStream& stream) final;
};

}  // namespace aoce