#include "FMediaMuxer.hpp"

#include <media/MediaHelper.hpp>

namespace aoce {
namespace ffmpeg {

FMediaMuxer::FMediaMuxer(/* args */) {
    videoEncoder = std::make_unique<FH264Encoder>();
    audioEncoder = std::make_unique<FAACEncoder>();
}

FMediaMuxer::~FMediaMuxer() {}

bool FMediaMuxer::start() {
    std::unique_lock<std::mutex> lck(mtx);
    if (bVideo) {
        bVideo = videoEncoder->configure(videoStream);
    }
    if (bAudio) {
        bAudio = audioEncoder->configure(audioStream);
    }
    if (!bAudio && !bVideo) {
        return false;
    }
    bRtmp = mediaType == MediaSourceType::rtmp;
    std::string formatName = getAvformat(url);
    //打开一个format_name类型的FormatContext
    AVOutputFormat* outputFormat = av_guess_format(
        bRtmp ? "flv" : formatName.c_str(), url.c_str(), nullptr);
    AVFormatContext* tempOut = nullptr;
    int32_t ret = avformat_alloc_output_context2(&tempOut, outputFormat,
                                                 nullptr, url.c_str());
    if (ret < 0) {
        std::string msg = "url:" + url + "could not open";
        logFRet(ret, msg);
        return false;
    }
    fmtCtx = getUniquePtr(tempOut);
    AVStream* stream = nullptr;
    if (bVideo) {
        stream = avformat_new_stream(fmtCtx.get(),
                                     videoEncoder->getCodecCtx()->codec);
        if (stream) {
            avcodec_parameters_from_context(stream->codecpar,
                                            videoEncoder->getCodecCtx());
            if (fmtCtx->oformat->flags & AVFMT_GLOBALHEADER) {
                videoEncoder->getCodecCtx()->flags |=
                    AV_CODEC_FLAG_GLOBAL_HEADER;
            }
            stream->time_base = videoEncoder->getCodecCtx()->time_base;
            stream->codecpar->codec_tag = 0;
            videoIndex = stream->index;
        }
    }
    if (bAudio) {
        stream = avformat_new_stream(fmtCtx.get(),
                                     audioEncoder->getCodecCtx()->codec);
        if (stream) {
            avcodec_parameters_from_context(stream->codecpar,
                                            audioEncoder->getCodecCtx());
            stream->time_base = audioEncoder->getCodecCtx()->time_base;
            if (stream->codecpar->extradata_size == 0) {
                uint8_t* dsi2 = (uint8_t*)av_malloc(2);
                makeDsi(stream->codecpar->sample_rate,
                        stream->codecpar->channels, dsi2);
                stream->codecpar->extradata_size = 2;
                stream->codecpar->extradata = dsi2;
            }
            stream->codecpar->codec_tag = 0;
            audioIndex = stream->index;
        }
    }
    av_dump_format(fmtCtx.get(), 0, url.c_str(), 1);
    AVDictionary* dict = nullptr;
    av_dict_set(&dict, "rtsp_transport", "tcp", 0);
    av_dict_set(&dict, "muxdelay", "0.0", 0);
    // 如果是文件，需要这步
    if ((fmtCtx->oformat->flags & AVFMT_NOFILE) == 0) {
        ret = avio_open2(&fmtCtx->pb, url.c_str(), AVIO_FLAG_WRITE, nullptr,
                         &dict);
        if (ret < 0) {
            av_dict_free(&dict);
            logFRet(ret, "avio_open fail:");
            return false;
        }
    }
    // 经调试,在这后fmtCtx->streams[videoIndex]->time_base会被改变1/1000
    ret = avformat_write_header(fmtCtx.get(), &dict);
    if (ret < 0) {
        logFRet(ret, "could not write header");
        return false;
    }
    // av_dict_free(&dict);
    bStart = true;
    return true;
}

void FMediaMuxer::pushAudio(const AudioFrame& audioFrame) {
    std::unique_lock<std::mutex> lck(mtx);
    if (!bAudio || !bStart || audioIndex < 0) {
        return;
    }
    int32_t ret = audioEncoder->input(audioFrame);
    if (ret < 0) {
        return;
    }
    while (true) {
        AVPacket packet = {};
        av_init_packet(&packet);
        int32_t ret =
            avcodec_receive_packet(audioEncoder->getCodecCtx(), &packet);
        // 已经读完
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            logFRet(ret, "h264 avcodec_receive_packet error.");
            av_packet_unref(&packet);
            break;
        }
        // buildAdts(packet.size, audioBuffer.data(), cdeCtx->sample_rate,
        //           cdeCtx->channels);
        // memcpy(audioBuffer.data() + 7, packet.data, packet.size);
        if (fmtCtx->streams[audioIndex]->time_base.num != 0) {
            av_packet_rescale_ts(&packet,
                                 audioEncoder->getCodecCtx()->time_base,
                                 fmtCtx->streams[audioIndex]->time_base);
        }
        packet.stream_index = audioIndex;
        ret = av_interleaved_write_frame(fmtCtx.get(), &packet);
        if (ret < 0) {
            logFRet(ret, "error write audio frame");
        }
        av_packet_unref(&packet);
    }
}

void FMediaMuxer::pushVideo(const VideoFrame& videoFrame) {
    std::unique_lock<std::mutex> lck(mtx);
    if (!bVideo || !bStart || videoIndex < 0) {
        return;
    }
    videoFps.run();
    int32_t ret = videoEncoder->input(videoFrame);
    while (true) {
        AVPacket packet = {};
        av_init_packet(&packet);
        int32_t ret =
            avcodec_receive_packet(videoEncoder->getCodecCtx(), &packet);
        // 已经读完
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            logFRet(ret, "h264 avcodec_receive_packet error.");
            av_packet_unref(&packet);
            break;
        }
        videoRate.run(packet.size);
        // codeccontex->stream
        if (fmtCtx->streams[videoIndex]->time_base.num != 0) {
            av_packet_rescale_ts(&packet,
                                 videoEncoder->getCodecCtx()->time_base,
                                 fmtCtx->streams[videoIndex]->time_base);
        }
        packet.stream_index = videoIndex;     
        ret = av_interleaved_write_frame(fmtCtx.get(), &packet);
        if (ret < 0) {
            logFRet(ret, "error write video frame");
        }
        av_packet_unref(&packet);
    }
}

void FMediaMuxer::stop() {
    std::unique_lock<std::mutex> lck(mtx);
    if (fmtCtx && bStart) {
        av_write_trailer(fmtCtx.get());
    }
    fmtCtx.reset();
    videoIndex = -1;
    audioIndex = -1;
    bStart = false;
}

void FMediaMuxer::release() { stop(); }

}  // namespace ffmpeg
}  // namespace aoce
