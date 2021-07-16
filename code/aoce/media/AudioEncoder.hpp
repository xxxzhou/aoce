#pragma once

#include "../Aoce.hpp"

namespace aoce {

class ACOE_EXPORT AudioEncoder {
   protected:
    /* data */
    AudioStream stream = {};

   public:
    AudioEncoder(/* args */);
    virtual ~AudioEncoder();

   public:
    bool configure(const AudioStream& stream) {
        this->stream = stream;
        if (stream.bitrate == 0) {
            this->stream.bitrate = 48000;
        }
        return onPrepare();
    }

   protected:
    virtual bool onPrepare() = 0;

   public:
    virtual int32_t input(const AudioFrame& frame) = 0;
};

}  // namespace aoce