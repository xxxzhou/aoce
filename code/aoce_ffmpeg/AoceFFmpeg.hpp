#pragma once
#include <Aoce.hpp>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/opt.h"
#include "libswresample/swresample.h"
#ifdef __cplusplus
}
#endif

#define AOCE_AAC_BUFFER_MAX_SIZE 1024 * 16
#define AOCE_H264_BUFFER_MAX_SIZE 1024 * 1024

namespace aoce {
namespace ffmpeg {

// std::unique_ptr<HDC__, std::function<void(HDC)>>
//	target_hdc(GetWindowDC(hwnd), [=](HDC x) { ReleaseDC(hwnd, x); });

template <typename T>
inline void freefobj(T* val) {
    av_free(val);
}

//显式具体化 对应各个具体实现
template <>
inline void freefobj(AVCodecContext* val) {
    avcodec_close(val);
    avcodec_free_context(&val);
}

template <>
inline void freefobj(AVFormatContext* val) {
    /*	if (val->flags & AVFMT_NOFILE) {
                    avio_close(val->pb);
                    avformat_free_context(val);
            }*/
    avformat_close_input(&val);
}

template <>
inline void freefobj(AVFrame* val) {
    av_frame_free(&val);
}

template <>
inline void freefobj(SwrContext* val) {
    swr_free(&val);
}

template <>
inline void freefobj(AVIOContext* val) {
    av_freep(&val->buffer);
    av_free(val);
}
#define AOCE_UNIQUE_FUNCTION(CLASSTYPE) \
    typedef void (*free##CLASSTYPE)(CLASSTYPE*);

#define AOCE_UNIQUE_FFCLASS(CLASSTYPE) \
    AOCE_UNIQUE_FUNCTION(CLASSTYPE)    \
    typedef std::unique_ptr<CLASSTYPE, free##CLASSTYPE> FO##CLASSTYPE;

//使用decltype,普通的函数指针,相应类型做为类的字段会声明不了
// typedef std::unique_ptr<CLASSTYPE, decltype(freefobj<CLASSTYPE>)*>
// O##CLASSTYPE;
#define AOCE_UNIQUE_FCLASS(CLASSTYPE)                                   \
    typedef std::unique_ptr<CLASSTYPE, std::function<void(CLASSTYPE*)>> \
        O##CLASSTYPE;                                                   \
    inline O##CLASSTYPE getUniquePtr(CLASSTYPE* ptr) {                  \
        O##CLASSTYPE uptr(ptr, freefobj<CLASSTYPE>);                    \
        return uptr;                                                    \
    }

AOCE_UNIQUE_FCLASS(AVFormatContext)
AOCE_UNIQUE_FCLASS(AVCodecContext)
AOCE_UNIQUE_FCLASS(AVFrame)
AOCE_UNIQUE_FCLASS(SwrContext)
AOCE_UNIQUE_FCLASS(AVIOContext)

void logFRet(int32_t ret, const std::string msg);

std::string getRetErrorStr(int32_t ret);

int32_t getSRIndex(uint32_t frequency);

int32_t getFormatFormSampleFmt(const char** fmt, enum AVSampleFormat sampleFmt);

bool checkSampleFmt(AVCodec* codec, enum AVSampleFormat sampleFmt);

void buildAdts(int size, uint8_t* buffer, int samplerate, int channels);

void makeDsi(int frequencyInHz, int channelCount, uint8_t* dsi);

int64_t rescaleTs(int64_t val, AVCodecContext* context, AVRational new_base);

}  // namespace ffmpeg
}  // namespace aoce