#pragma once

#include "../Aoce.hpp"

namespace aoce {

enum class RateControl {
    none,
    cbr,
    vbr,
    qp,
};

// struct VideoCodecParamet {
//     // 码率控制
//     RateControl rateControl = RateControl::none;
// };

class ACOE_EXPORT VideoEncoder {
   protected:
    VideoStream stream = {};

   public:
    VideoEncoder(/* args */);
    virtual ~VideoEncoder();

   public:
    bool configure(const VideoStream& stream) {
        this->stream = stream;
        if (this->stream.fps == 0) {
            this->stream.fps = 30;
        }
        if (this->stream.bitrate == 0) {
            this->stream.bitrate =
                (stream.width / 10) * (stream.height / 10) * stream.fps * 10;
        }
        return onPrepare();
    }

   protected:
    virtual bool onPrepare() = 0;

   public:
    virtual int32_t input(const VideoFrame& frame) = 0;
};
}  // namespace aoce
