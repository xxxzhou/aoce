#pragma once

#include "../Aoce.hpp"

namespace aoce {

ACOE_EXPORT MediaSourceType getMediaType(const std::string& str);

class ACOE_EXPORT MediaPlayer : public IMediaPlayer {
   protected:
    IMediaPlayerObserver* observer;
    std::string uri = "";
    PlayStatus status = PlayStatus::idle;
    MediaSourceType mediaType = MediaSourceType::other;
    AudioStream audioStream = {};
    VideoStream videoStream = {};

   public:
    MediaPlayer(/* args */){};
    virtual ~MediaPlayer(){};

   public:
    // 同步的prepare需要之后才能拿到,异步在回调里的onPrepared
    virtual const AudioStream& getAudioStream() final { return audioStream; };
    // 同步的prepare需要之后才能拿到,异步在回调里的onPrepared
    virtual const VideoStream& getVideoStream() final { return videoStream; };

   protected:
    virtual void onSetObserver(){};
    virtual void onSetDataSource(){};

   public:
    virtual void setObserver(IMediaPlayerObserver* observer) final;
    // 文件路径,URL(RTMP这些)
    virtual void setDataSource(const char* path) final;
    // virtual void prepare(bool bAsync) = 0;
    // 同步prepare可以在下面直接调用start,否则需要在observer里的prepare调用
    // virtual void start() = 0;
    virtual void pause(){};
    // virtual void stop() = 0;
    virtual void release(){};
};

}  // namespace aoce