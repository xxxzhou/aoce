#pragma once

#include <Media/MediaPlayer.hpp>
#include <mutex>

#include "AoceFFmpeg.hpp"

namespace aoce {
namespace ffmpeg {

// https://www.cnblogs.com/zhouxin/p/12651304.html
// https://www.cnblogs.com/rustfisher/p/11554873.html
class FMediaPlayer : public MediaPlayer {
   private:
    OAVFormatContext fmtCtx = nullptr;
    OAVCodecContext audioCtx = nullptr;
    OAVCodecContext videoCtx = nullptr;
    OAVFrame frame = nullptr;
    OSwrContext swrCtx = nullptr;
    //音频数据输出格式
    AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_S16;
    int32_t outLayout = AV_CH_LAYOUT_MONO;

    int32_t videoIndex = -1;
    int32_t audioIndex = -1;
    uint64_t audioTimestamp = 0;
    uint64_t videoTimestamp = 0;

    std::mutex mtx;
    std::condition_variable preSignal;

   public:
    FMediaPlayer(/* args */);
    virtual ~FMediaPlayer() override;
    friend int decode_interrupt_cb(void* ctx);

   protected:
    void onError(PlayStatus status, const std::string& msg, int32_t ret = 0);

   public:
    virtual void prepare(bool bAsync) override;
    virtual void start() override;
    virtual void stop() override;
    virtual void release() override;
};
}  // namespace ffmpeg

}  // namespace aoce