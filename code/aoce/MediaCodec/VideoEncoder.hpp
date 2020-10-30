#pragma once

#include "../Aoce.hpp"

namespace aoce {

struct EncoderOut {
    int8_t* data = nullptr;
    int32_t lenght = 0;
    int64_t timeStamp = 0;
};

enum class VideoRateControl {
    none,
    cbr,
    vbr,
    qp,
};

struct VideoEncoderParamet {
    VideoStream stream = {};
    VideoRateControl rateControl = VideoRateControl::none;
};

class VideoEncoder {
   protected:
    VideoEncoderParamet paramet;

   public:
    VideoEncoder(/* args */){};
    virtual ~VideoEncoder(){};

   public:
    virtual void configure(const VideoEncoderParamet& paramet) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;

   public:
    virtual void input(const VideoFrame& frame) = 0;
    virtual void output(const EncoderOut& packet) = 0;
};
}  // namespace aoce
