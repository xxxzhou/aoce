#include "FMediaPlayer.hpp"

#include <thread>
namespace aoce {
namespace ffmpeg {

int decode_interrupt_cb(void* ctx) {
    FMediaPlayer* player = static_cast<FMediaPlayer*>(ctx);
    if (!player) {
        return 1;
    }
    if (player->status == PlayStatus::stopped) {
        return 1;  // 1 AVERROR_EOF abort
    }
    return 0;
}

FMediaPlayer::FMediaPlayer(/* args */) {}

FMediaPlayer::~FMediaPlayer() { release(); }

void FMediaPlayer::onError(PlayStatus status, const std::string& msg,
                           int32_t ret) {
    if (ret != 0) {
        logFFmpegRet(ret, msg);
    }
    if (observer) {
        observer->onError(status, ret, msg);
    }
    this->status = PlayStatus::error;
}

void FMediaPlayer::prepare(bool bAsync) {
    if (status == PlayStatus::preparing || status == PlayStatus::prepared) {
        logMessage(LogLevel::info, "media play in preparing");
        return;
    }
    logAssert(status == PlayStatus::init || status == PlayStatus::stopped,
              "media play current status is not init");
    status = PlayStatus::preparing;
    int32_t ret = 0;
    auto preMedia = [&]() {
        AVFormatContext* temp = avformat_alloc_context();
        temp->interrupt_callback.callback = decode_interrupt_cb;
        temp->interrupt_callback.opaque = this;
        AVDictionary* dict = nullptr;
        // av_dict_set(&dict, "rw_timeout", "3000000", 0);
        // av_dict_set(&dict, "max_delay", "3000000", 0);
        // av_dict_set(&dict, "rtsp_transport", "tcp", 0);  //采用tcp传输
        // av_dict_set(&dict, "stimeout", "2000000", 0);
        if ((ret = avformat_open_input(&temp, uri.c_str(), 0, &dict)) < 0) {
            avformat_free_context(temp);
            av_dict_free(&dict);
            onError(PlayStatus::preparing,
                    "media play avformat open input error.", ret);
            return;
        }
        fmtCtx = getUniquePtr(temp);
        if ((ret = avformat_find_stream_info(fmtCtx.get(), nullptr)) < 0) {
            onError(PlayStatus::preparing, "media play find stream fail.", ret);
            return;
        }
        for (int32_t i = 0; i < fmtCtx->nb_streams; i++) {
            auto st = fmtCtx->streams[i];
            if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoIndex = i;
                auto codec = avcodec_find_decoder(st->codecpar->codec_id);
                auto temp = avcodec_alloc_context3(codec);
                if (temp == nullptr) {
                    onError(PlayStatus::preparing,
                            "media play alloc video context error.");
                    return;
                }
                videoCtx = getUniquePtr(temp);
                // 如fmtCtx->streams[videoIndex]->codecpar里包含了AV_PIX_FMT_YUV422P数据
                avcodec_parameters_to_context(
                    videoCtx.get(), fmtCtx->streams[videoIndex]->codecpar);
                if ((ret = avcodec_open2(videoCtx.get(), codec, nullptr)) < 0) {
                    onError(PlayStatus::preparing,
                            "media play cannot open video decoder.", ret);
                    return;
                }
                videoStream.width = st->codecpar->width;
                videoStream.height = st->codecpar->height;
                videoStream.fps = av_q2intfloat(videoCtx->framerate);
                if (videoCtx->pix_fmt == AV_PIX_FMT_YUV420P) {
                    videoStream.videoType = VideoType::yuv420P;
                } else if (videoCtx->pix_fmt == AV_PIX_FMT_YUV422P) {
                    videoStream.videoType = VideoType::yuy2P;
                }
                videoStream.bitrate = videoCtx->bit_rate;
            } else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioIndex = i;
                auto codec = avcodec_find_decoder(
                    fmtCtx->streams[audioIndex]->codecpar->codec_id);
                auto temp = avcodec_alloc_context3(codec);
                if (!temp) {
                    if (observer) {
                        onError(PlayStatus::preparing,
                                "media play alloc audio context error.");
                    }
                    this->status = PlayStatus::error;
                    return;
                }
                audioCtx = getUniquePtr(temp);
                // 如fmtCtx->streams[videoIndex]->codecpar里包含了AV_PIX_FMT_YUV422P数据
                avcodec_parameters_to_context(
                    audioCtx.get(), fmtCtx->streams[audioIndex]->codecpar);
                if ((ret = avcodec_open2(audioCtx.get(), codec, nullptr)) < 0) {
                    onError(PlayStatus::preparing,
                            "media play cannot open audio decoder.", ret);
                    return;
                }
                // 分配音频重采样(原始数据格式如果是平面格式，转化成交叉格式与单通)
                auto tempSwr = swr_alloc_set_opts(
                    nullptr, outLayout, outSampleFormat, audioCtx->sample_rate,
                    av_get_default_channel_layout(audioCtx->channels),
                    audioCtx->sample_fmt, audioCtx->sample_rate, 0, nullptr);
                if (!tempSwr) {
                    onError(PlayStatus::preparing,
                            "media play could not allocate resampler context.");
                    return;
                }
                swrCtx = getUniquePtr(tempSwr);
                int32_t ret = swr_init(swrCtx.get());
                if (ret < 0) {
                    onError(PlayStatus::preparing,
                            "failed to initialize the resampling context!",
                            ret);
                    return;
                }
                audioStream.depth = 16;
                audioStream.bitrate = audioCtx->bit_rate;
                audioStream.sampleRate = audioCtx->sample_rate;
                audioStream.channel =
                    av_get_channel_layout_nb_channels(outLayout);
            }
        }
        status = PlayStatus::prepared;
        observer->onPrepared();
    };
    if (bAsync) {
        std::thread preThread(preMedia);
        preThread.detach();
    } else {
        preMedia();
    }
}

void FMediaPlayer::start() {
    // 已经开始了
    if (status == PlayStatus::started) {
        return;
    }
    logAssert(status == PlayStatus::prepared,
              "media play current status is not prepared");
    std::thread runThread([&]() {
        status = PlayStatus::started;
        int32_t ret = 0;
        frame = getUniquePtr(av_frame_alloc());
        // 声音转码
        OAVFrame aframe = getUniquePtr(av_frame_alloc());
        if (audioIndex >= 0) {
            aframe->nb_samples = audioCtx->frame_size;
            aframe->channel_layout = outLayout;
            aframe->format = outSampleFormat;
            aframe->sample_rate = audioCtx->sample_rate;
            if ((ret = av_frame_get_buffer(aframe.get(), 0)) < 0) {
                onError(PlayStatus::started,
                        "error allocating an audio buffer.", ret);
                return;
            }
        }
        int64_t minPts = INT64_MAX;
        int64_t minNowMs = 0;
        while (status == PlayStatus::started) {
            AVPacket packet;
            av_init_packet(&packet);
            // av_read_frame本身会阻塞当前线程(可能以分钟记),故以等待加状态改变在这无用
            if ((ret = av_read_frame(fmtCtx.get(), &packet)) < 0) {
                break;
            }
            if (packet.stream_index == videoIndex) {
                ret = avcodec_send_packet(videoCtx.get(), &packet);
                if (ret < 0) {
                    onError(PlayStatus::started,
                            "media play video avcodec_send_packet fail", ret);
                    av_packet_unref(&packet);
                    break;
                }
                // videoRate.run(packet.size);
                while (ret >= 0) {
                    ret = avcodec_receive_frame(videoCtx.get(), frame.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        break;
                    }
                    // 从当前contex以timebase为底的时间转化成毫秒
                    frame->pts = av_rescale_q(
                        frame->best_effort_timestamp,
                        fmtCtx->streams[packet.stream_index]->time_base,
                        av_make_q(1, 1000));
                    if (observer) {
                        VideoFrame videoFrame = {};
                        videoFrame.width = frame->width;
                        videoFrame.height = frame->height;
                        videoFrame.data[0] = frame->data[0];
                        videoFrame.data[1] = frame->data[1];
                        videoFrame.data[2] = frame->data[2];
                        videoFrame.videoType = videoStream.videoType;
                        videoFrame.timeStamp = frame->pts;
                        memcpy(videoFrame.dataAlign, frame->linesize,
                               sizeof(videoFrame.dataAlign));
                        observer->onVideoFrame(videoFrame);
                    }
                    if (mediaType == MediaType::file) {
                        if (frame->pts < minPts) {
                            minPts = frame->pts;
                            minNowMs = getNowTimeStamp();
                        }
                        int64_t sc = (frame->pts - minPts) -
                                     (getNowTimeStamp() - minNowMs);
                        if (sc > 0) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(sc));
                        }
                    }
                }
                av_packet_unref(&packet);
            } else if (packet.stream_index == audioIndex) {
                ret = avcodec_send_packet(audioCtx.get(), &packet);
                if (ret < 0) {
                    onError(PlayStatus::started,
                            "media play audio avcodec_send_packet fail", ret);
                    av_packet_unref(&packet);
                    break;
                }
                while (ret >= 0) {
                    ret = avcodec_receive_frame(audioCtx.get(), frame.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        break;
                    }
                    frame->pts = av_rescale_q(
                        frame->best_effort_timestamp,
                        fmtCtx->streams[packet.stream_index]->time_base,
                        av_make_q(1, 1000));
                    int32_t dataSize =
                        aframe->nb_samples *
                        av_get_channel_layout_nb_channels(outLayout) *
                        av_get_bytes_per_sample(outSampleFormat);
                    if (observer) {
                        swr_convert(swrCtx.get(), (uint8_t**)aframe->data,
                                    aframe->nb_samples,
                                    (const uint8_t**)frame->data,
                                    frame->nb_samples);
                        AudioFrame audioFrame = {};
                        audioFrame.depth =
                            av_get_bytes_per_sample(outSampleFormat) * 8;
                        audioFrame.channel =
                            av_get_channel_layout_nb_channels(outLayout);
                        audioFrame.data[0] = aframe->data[0];
                        audioFrame.data[1] = aframe->data[1];
                        audioFrame.dataSize = dataSize;
                        audioFrame.sampleRate = audioCtx->sample_rate;
                        audioFrame.timeStamp = frame->pts;
                        observer->onAudioFrame(audioFrame);
                    }
                    if (mediaType == MediaType::file) {
                        if (frame->pts < minPts) {
                            minPts = frame->pts;
                            minNowMs = getNowTimeStamp();
                        }
                        int64_t sc = (frame->pts - minPts) -
                                     (getNowTimeStamp() - minNowMs);
                        if (sc > 0) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(sc));
                        }
                    }
                }
            }
        }
        status = PlayStatus::completed;
        // 发送一个信号
        preSignal.notify_all();
        if (observer) {
            observer->onComplate();
        }
    });
    runThread.detach();
}

void FMediaPlayer::stop() {
    if (status == PlayStatus::stopped) {
        return;
    }
    PlayStatus preStatus = status;
    status = PlayStatus::stopped;
    if (preStatus == PlayStatus::started) {
        std::unique_lock<std::mutex> lck(mtx);
        // 等待preSignal信号回传
        auto status = preSignal.wait_for(lck, std::chrono::seconds(2));
        if (status == std::cv_status::timeout) {
            logMessage(LogLevel::warn, "media play close time out.");
        }
    }
}

void FMediaPlayer::release() {
    if (status == PlayStatus::end) {
        return;
    }
    if (status == PlayStatus::started) {
        stop();
    }
    fmtCtx.reset();
    audioCtx.reset();
    videoCtx.reset();
    status = PlayStatus::end;
}

}  // namespace ffmpeg
}  // namespace aoce