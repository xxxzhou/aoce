#pragma once

#include "../Aoce.hpp"

namespace aoce {

enum class PlayStatus {
    idle,
    end,
    error,
    init,
    preparing,
    prepared,
    started,
    stopped,
    paused,
    completed,
};

enum class MediaType {
    other,
    file,
    rtmp,
    http,
};

class ACOE_EXPORT IMediaPlayerObserver {
   public:
    IMediaPlayerObserver(/* args */){};
    virtual ~IMediaPlayerObserver() {};

   public:
    // prepared后,能拿到MediaPlayer的视频与音频信息
    virtual void onPrepared(){};
    virtual void onError(PlayStatus staus, int32_t code, std::string msg){};
    virtual void onVideoFrame(const VideoFrame& frame){};
    virtual void onAudioFrame(const AudioFrame& frame){};
    virtual void onStop(){};
    virtual void onComplate(){};
};

ACOE_EXPORT MediaType getMediaType(const std::string& str);

class ACOE_EXPORT MediaPlayer {
   protected:
    IMediaPlayerObserver* observer;
    std::string uri = "";
    PlayStatus status = PlayStatus::idle;
    MediaType mediaType = MediaType::other;
    AudioStream audioStream = {};
    VideoStream videoStream = {};

   public:
    MediaPlayer(/* args */){};
    virtual ~MediaPlayer(){};

   protected:
    virtual void onSetObserver(){};
    virtual void onSetDataSource(){};

   public:
    void setObserver(IMediaPlayerObserver* observer);
    // 文件路径,URL(RTMP这些)
    void setDataSource(const std::string& path);
    virtual void prepare(bool bAsync) = 0;
    // 同步prepare可以在下面直接调用start,否则需要在observer里的prepare调用
    virtual void start() = 0;
    virtual void pause(){};
    virtual void stop() = 0;
    virtual void release(){};
};

}  // namespace aoce