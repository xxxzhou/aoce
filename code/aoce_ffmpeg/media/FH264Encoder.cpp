#include "FH264Encoder.hpp"

namespace aoce {
namespace ffmpeg {

FH264Encoder::FH264Encoder(/* args */) {}

FH264Encoder::~FH264Encoder() {}

bool FH264Encoder::onPrepare() {
    std::vector<std::string> videoEncodes = {"libx264", "h264_nvenc",
                                             "h264_qsv"};
    for (const auto& name : videoEncodes) {
        auto codec = avcodec_find_encoder_by_name(name.c_str());
        if (!codec) {
            continue;
        }
        AVCodecContext* temp = avcodec_alloc_context3(codec);
        if (!temp) {
            continue;
        }
        cdeCtx = getUniquePtr(temp);
        cdeCtx->codec_type = AVMEDIA_TYPE_VIDEO;
        cdeCtx->width = stream.width;
        cdeCtx->height = stream.height;
        cdeCtx->coded_width = stream.width;
        cdeCtx->coded_height = stream.height;
        cdeCtx->bit_rate = stream.bitrate;
        cdeCtx->rc_buffer_size = stream.bitrate;
        cdeCtx->gop_size = stream.fps * 1;
        cdeCtx->time_base = {1, stream.fps};
        cdeCtx->delay = 0;
        // H264
        if (rate == RateControl::vbr) {  // VBR
            cdeCtx->flags |= AV_CODEC_FLAG_PASS1;
            cdeCtx->flags |= AV_CODEC_FLAG_QSCALE;
            cdeCtx->rc_min_rate = stream.bitrate;
            cdeCtx->rc_max_rate = stream.bitrate * 3 / 2;
        } else if (rate == RateControl::qp) {  // QP
            cdeCtx->qmin = 21;
            cdeCtx->qmax = 26;
        }
        cdeCtx->has_b_frames = 0;
        cdeCtx->max_b_frames = 0;
        cdeCtx->me_pre_cmp = 2;
        AVDictionary* param = nullptr;
        if (stream.videoType == VideoType::yuv420P) {
            temp->pix_fmt = AV_PIX_FMT_YUV420P;
            av_dict_set(&param, "profile", "high", 0);
        } else if (stream.videoType == VideoType::yuy2P) {
            temp->pix_fmt = AV_PIX_FMT_YUV422P;
            av_dict_set(&param, "profile", "high422", 0);
        }
        if (cdeCtx->codec_id == AV_CODEC_ID_H264) {
            av_dict_set(&param, "tune", "zerolatency", 0);
        }
        int ret = avcodec_open2(cdeCtx.get(), codec, &param);
        if (ret < 0) {
            cdeCtx.reset();
            logFRet(ret, "open h264 codec failed.");
            continue;
        }
        break;
    }
    if (!cdeCtx) {
        return false;
    }
    frame = getUniquePtr(av_frame_alloc());
    if (!frame) {
        return false;
    }
    frame->format = cdeCtx->pix_fmt;
    frame->width = cdeCtx->width;
    frame->height = cdeCtx->height;
    frame->linesize[0] = cdeCtx->width;
    frame->linesize[1] = cdeCtx->width / 2;
    frame->linesize[2] = cdeCtx->width / 2;
    totalFrame = 0;
    return true;
}

int32_t FH264Encoder::input(const VideoFrame& vframe) {
    frame->data[0] = vframe.data[0];
    frame->data[1] = vframe.data[1];
    frame->data[2] = vframe.data[2];
    if (vframe.dataAlign[0] > 0) {
        frame->linesize[0] = vframe.dataAlign[0];
    }
    if (vframe.dataAlign[1] > 0) {
        frame->linesize[1] = vframe.dataAlign[1];
    }
    if (vframe.dataAlign[2] > 0) {
        frame->linesize[2] = vframe.dataAlign[2];
    }
    frame->pts = totalFrame;
    int32_t ret = avcodec_send_frame(cdeCtx.get(), frame.get());
    if (ret < 0) {
        logFRet(ret, "h264 avcodec_send_frame error.");
    }
    totalFrame++;
    return ret;
}

// int32_t FH264Encoder::output(VideoPacket& vpacket) {
//     AVPacket packet = {};
//     av_init_packet(&packet);
//     int32_t ret = avcodec_receive_packet(cdeCtx.get(), &packet);
//     if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
//         return 0;
//     } else if (ret < 0) {
//         logFRet(ret, "h264 avcodec_receive_packet error.");
//         av_packet_unref(&packet);
//         return ret;
//     }
//     if (packet.size > AOCE_H264_BUFFER_MAX_SIZE) {
//         return -2;
//     }
//     memcpy(vpacket.data, packet.data, packet.size);
//     vpacket.lenght = packet.size;
//     vpacket.timeStamp = packet.pts;
//     av_packet_unref(&packet);
//     return 0;
// }

}  // namespace ffmpeg
}  // namespace aoce