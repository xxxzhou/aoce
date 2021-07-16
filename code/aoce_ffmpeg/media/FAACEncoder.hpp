#pragma once

#include <media/AudioEncoder.hpp>
#include <mutex>

#include "../AoceFFmpeg.hpp"

namespace aoce {
namespace ffmpeg {

class FAACEncoder : public AudioEncoder {
   private:
    /* data */
    OAVCodecContext cdeCtx = nullptr;
    OAVFrame frame = nullptr;
    int32_t bufferSize = 0;
    OSwrContext swrCtx = nullptr;
    // 限定音频数据输入格式
    AVSampleFormat inSampleFormat = AV_SAMPLE_FMT_S16;
    // 指定AAC输出数据格式
    AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_FLTP;
    uint8_t* samples = nullptr;
    std::vector<uint8_t> pcmBuffer;
    uint32_t pcmOffset = 0;
    int64_t totalFrame = 0;

   public:
    FAACEncoder(/* args */);
    ~FAACEncoder();

   public:
    AVCodecContext* getCodecCtx() { return cdeCtx.get(); }

   protected:
    virtual bool onPrepare() override;

   public:
    virtual int32_t input(const AudioFrame& aframe) override;
};

}  // namespace ffmpeg
}  // namespace aoce