#pragma once

#include <media/VideoEncoder.hpp>
#include <mutex>

#include "../AoceFFmpeg.hpp"

namespace aoce {
namespace ffmpeg {

class FH264Encoder : public VideoEncoder {
   private:
    /* data */
    OAVCodecContext cdeCtx = nullptr;
    OAVFrame frame = nullptr;
    RateControl rate = RateControl::vbr;
    int64_t totalFrame = 0;

   public:
    FH264Encoder(/* args */);
    virtual ~FH264Encoder();

   public:
    AVCodecContext* getCodecCtx() { return cdeCtx.get(); }

   protected:
    virtual bool onPrepare() override;

   public:
    virtual int32_t input(const VideoFrame& frame) override;

};

}  // namespace ffmpeg
}  // namespace aoce