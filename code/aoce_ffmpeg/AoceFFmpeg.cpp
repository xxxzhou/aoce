#include "AoceFFmpeg.hpp"

namespace aoce {
namespace ffmpeg {

void logFRet(int32_t ret, const std::string msg) {
    char error_char[AV_ERROR_MAX_STRING_SIZE];
    std::string message =
        msg + " ret: " +
        av_make_error_string(error_char, AV_ERROR_MAX_STRING_SIZE, ret);
    logMessage(ret >= 0 ? LogLevel::info : LogLevel::error, message.c_str());
}

std::string getRetErrorStr(int32_t ret) {
    char error_char[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(error_char, AV_ERROR_MAX_STRING_SIZE, ret);
    std::string msg;
    string_format(msg, " ret:", ret, " ", error_char);
    return msg;
}

int32_t getSRIndex(uint32_t frequency) {
    switch (frequency) {
        case 96000:
            return 0;
        case 88200:
            return 1;
        case 64000:
            return 2;
        case 48000:
            return 3;
        case 44100:
            return 4;
        case 32000:
            return 5;
        case 24000:
            return 6;
        case 22050:
            return 7;
        case 16000:
            return 8;
        case 12000:
            return 9;
        case 11025:
            return 10;
        case 8000:
            return 11;
        case 7350:
            return 12;
        default:
            return 0;
    }
    return 0;
}

int32_t getFormatFormSampleFmt(const char** fmt,
                               enum AVSampleFormat sampleFmt) {
    struct sample_fmt_entry {
        enum AVSampleFormat sampleFmt;
        const char *fmtBe, *fmtLe;
    } sampleFmtEntries[] = {
        {AV_SAMPLE_FMT_U8, "u8", "u8"},
        {AV_SAMPLE_FMT_S16, "s16be", "s16le"},
        {AV_SAMPLE_FMT_S32, "s32be", "s32le"},
        {AV_SAMPLE_FMT_FLT, "f32be", "f32le"},
        {AV_SAMPLE_FMT_DBL, "f64be", "f64le"},
    };
    *fmt = nullptr;

    for (int32_t i = 0; i < FF_ARRAY_ELEMS(sampleFmtEntries); i++) {
        struct sample_fmt_entry* entry = &sampleFmtEntries[i];
        if (sampleFmt == entry->sampleFmt) {
            *fmt = AV_NE(entry->fmtBe, entry->fmtLe);
            return 0;
        }
    }
    std::string msg;
    string_format(msg, "sample format ", av_get_sample_fmt_name(sampleFmt),
                  " not supported");
    logMessage(LogLevel::error, msg);
    return AVERROR(EINVAL);
}

bool checkSampleFmt(AVCodec* codec, enum AVSampleFormat sampleFmt) {
    const enum AVSampleFormat* p = codec->sample_fmts;
    int i = 0;
    while (p[i] != AV_SAMPLE_FMT_NONE) {
        if (p[i] == sampleFmt) {
            return true;
        }
        i++;
    }
    return false;
}

void buildAdts(int size, uint8_t* buffer, int samplerate, int channels) {
    char* padts = (char*)buffer;
    int profile = 2;                       // AAC LC
    int freqIdx = getSRIndex(samplerate);  // 44.1KHz
    int chanCfg =
        channels;  // MPEG-4 Audio Channel Configuration. 1 Channel front-center
    padts[0] = (char)0xFF;  // 11111111     = syncword
    padts[1] = (char)0xF1;  // 1111 1 00 1  = syncword MPEG-2 Layer CRC
    padts[2] = (char)(((profile - 1) << 6) + (freqIdx << 2) + (chanCfg >> 2));
    padts[6] = (char)0xFC;
    padts[3] = (char)(((chanCfg & 3) << 6) + ((7 + size) >> 11));
    padts[4] = (char)(((7 + size) & 0x7FF) >> 3);
    padts[5] = (char)((((7 + size) & 7) << 5) + 0x1F);
}

void makeDsi(int frequencyInHz, int channelCount, uint8_t* dsi) {
    int sampling_frequency_index = getSRIndex(frequencyInHz);
    unsigned int object_type = 2;  // AAC LC by default
    dsi[0] = (object_type << 3) | (sampling_frequency_index >> 1);
    dsi[1] = ((sampling_frequency_index & 1) << 7) | (channelCount << 3);
}

int64_t rescaleTs(int64_t val, AVCodecContext* context, AVRational new_base) {
    return av_rescale_q_rnd(
        val, context->time_base, new_base,
        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
}

}  // namespace ffmpeg
}  // namespace aoce