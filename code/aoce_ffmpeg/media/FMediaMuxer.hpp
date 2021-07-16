#pragma once

#include <media/MediaMuxer.hpp>
#include <mutex>

#include "../AoceFFmpeg.hpp"
#include "FAACEncoder.hpp"
#include "FH264Encoder.hpp"

namespace aoce {
namespace ffmpeg {

class FMediaMuxer : public MediaMuxer {
   private:
    OAVFormatContext fmtCtx = nullptr;
    /* data */
    std::unique_ptr<FH264Encoder> videoEncoder = nullptr;
    std::unique_ptr<FAACEncoder> audioEncoder = nullptr;
    int32_t videoIndex = -1;
    int32_t audioIndex = -1;
    std::mutex mtx;
    bool bStart = false;
    bool bRtmp = false;

   public:
    FMediaMuxer(/* args */);
    ~FMediaMuxer();

   public:
    virtual bool start() override;
    virtual void pushAudio(const AudioFrame& audioFrame) override;
    virtual void pushVideo(const VideoFrame& videoFrame) override;
    virtual void stop() override;
    virtual void release() override;
};

}  // namespace ffmpeg
}  // namespace aoce