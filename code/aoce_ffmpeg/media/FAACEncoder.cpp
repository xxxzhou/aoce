#include "FAACEncoder.hpp"

namespace aoce {
namespace ffmpeg {

FAACEncoder::FAACEncoder(/* args */) {}

FAACEncoder::~FAACEncoder() {
    if (samples) {
        av_free(samples);
        samples = nullptr;
    }
}

bool FAACEncoder::onPrepare() {
    auto codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!codec) {
        logMessage(LogLevel::warn, "aoce_ffmpeg could not find acc encoder.");
        return false;
    }
    AVCodecContext* temp = avcodec_alloc_context3(codec);
    if (!temp) {
        logMessage(LogLevel::warn,
                   "aoce_ffmpeg avcodec_alloc_context3 failed.");
        return false;
    }
    //设置ACC编码属性
    cdeCtx = getUniquePtr(temp);
    cdeCtx->profile = FF_PROFILE_AAC_LOW;
    cdeCtx->codec_type = AVMEDIA_TYPE_AUDIO;
    cdeCtx->bit_rate = stream.bitrate;
    cdeCtx->time_base = {1, stream.sampleRate};
    cdeCtx->sample_fmt = outSampleFormat;
    cdeCtx->sample_rate = stream.sampleRate;
    cdeCtx->channel_layout = stream.channel == 1
                                 ? AV_CH_LAYOUT_MONO
                                 : AV_CH_LAYOUT_STEREO;  // 双声道
    cdeCtx->channels =
        av_get_channel_layout_nb_channels(cdeCtx->channel_layout);
    if (!checkSampleFmt(codec, cdeCtx->sample_fmt)) {
        logMessage(
            LogLevel::warn,
            "acc encoder does not support sample format AV_SAMPLE_FMT_FLTP");
        return false;
    }
    if (avcodec_open2(cdeCtx.get(), codec, nullptr) < 0) {
        logMessage(LogLevel::warn, "acc avcodec_open2 failed!");
        return false;
    }
    //分配音频桢
    frame = getUniquePtr(av_frame_alloc());
    frame->nb_samples = cdeCtx->frame_size;
    frame->format = cdeCtx->sample_fmt;
    frame->channel_layout = cdeCtx->channel_layout;
    frame->channels = cdeCtx->channels;
    frame->sample_rate = cdeCtx->sample_rate;
    //分配音频重采样(原始数据格式转化成AAC所需要数据格式)
    auto tempSwr = swr_alloc_set_opts(
        nullptr, av_get_default_channel_layout(cdeCtx->channels),
        cdeCtx->sample_fmt, cdeCtx->sample_rate,
        av_get_default_channel_layout(cdeCtx->channels), inSampleFormat,
        cdeCtx->sample_rate, 0, nullptr);
    if (!tempSwr) {
        logMessage(LogLevel::warn, "could not allocate resampler context!");
        return false;
    }
    swrCtx = getUniquePtr(tempSwr);
    int32_t ret = swr_init(swrCtx.get());
    if (ret < 0) {
        logFRet(ret, "failed to initialize the resampling context!");
        return false;
    }
    bufferSize = av_samples_get_buffer_size(
        nullptr, cdeCtx->channels, cdeCtx->frame_size, cdeCtx->sample_fmt, 0);
    if (bufferSize < 0) {
        logMessage(LogLevel::warn, "acc av_samples_get_buffer_size failed!");
        return false;
    }
    samples = static_cast<uint8_t*>(av_malloc(bufferSize));
    ret = avcodec_fill_audio_frame(
        frame.get(), cdeCtx->channels, cdeCtx->sample_fmt,
        reinterpret_cast<const uint8_t*>(samples), bufferSize, 0);
    if (ret < 0) {
        logFRet(ret, "acc avcodec_fill_audio_frame failed!");
        return ret;
    }
    pcmBuffer.resize(AOCE_AAC_BUFFER_MAX_SIZE);
    totalFrame = 0;
    return true;
}

int32_t FAACEncoder::input(const AudioFrame& aframe) {
    memcpy(pcmBuffer.data() + pcmOffset, aframe.data[0], aframe.dataSize);
    pcmOffset += aframe.dataSize;
    int32_t frameSize = cdeCtx->frame_size * cdeCtx->channels *
                        av_get_bytes_per_sample(inSampleFormat);
    // 每次处理framesize数据
    while (pcmOffset >= frameSize) {
        pcmOffset -= frameSize;
        frame->data[0] = const_cast<uint8_t*>(pcmBuffer.data());
        OAVFrame cframe = getUniquePtr(av_frame_alloc());
        cframe->nb_samples = cdeCtx->frame_size;
        cframe->format = cdeCtx->sample_fmt;
        cframe->channel_layout = cdeCtx->channel_layout;
        cframe->channels =
            av_get_channel_layout_nb_channels(cdeCtx->channel_layout);
        cframe->sample_rate = cdeCtx->sample_rate;
        int ret = av_frame_get_buffer(cframe.get(), 0);
        if (ret < 0) {
            logFRet(ret, "error allocating an audio buffer.");
            return ret;
        }
        swr_convert(swrCtx.get(), (uint8_t**)cframe->data, cframe->nb_samples,
                    (const uint8_t**)frame->data, frame->nb_samples);
        cframe->linesize[0] = frameSize;
        cframe->linesize[1] = frameSize;
        cframe->pts = av_rescale_q(totalFrame, {1, cdeCtx->sample_rate},
                                   cdeCtx->time_base);
        ret = avcodec_send_frame(cdeCtx.get(), cframe.get());
        if (ret < 0) {
            logFRet(ret, "aac avcodec_send_frame error.");
            return ret;
        }
        memmove(pcmBuffer.data(), pcmBuffer.data() + frameSize, pcmOffset);
        totalFrame += cdeCtx->frame_size;
    }
    return 0;
}

// int32_t FAACEncoder::output(AudioPacket& apacket) {
//     AVPacket packet = {};
//     av_init_packet(&packet);
//     int32_t ret = avcodec_receive_packet(cdeCtx.get(), &packet);
//     if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
//         return 0;
//     } else if (ret < 0) {
//         logFRet(ret, "acc avcodec_receive_packet error.");
//         av_packet_unref(&packet);
//         return ret;
//     }
//     if (packet.size > AOCE_AAC_BUFFER_MAX_SIZE) {
//         return -2;
//     }
//     buildAdts(packet.size, apacket.data, cdeCtx->sample_rate,
//     cdeCtx->channels); memcpy(apacket.data + 7, packet.data, packet.size);
//     apacket.lenght = packet.size + 7;
//     apacket.timeStamp = packet.pts;
//     av_packet_unref(&packet);
//     return 0;
// }

}  // namespace ffmpeg
}  // namespace aoce