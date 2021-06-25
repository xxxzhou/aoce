#pragma once
#include "Aoce.h"

namespace aoce {

enum class PlayStatus : int32_t {
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

enum class MediaSourceType : int32_t {
    other,
    file,
    rtmp,
    http,
};

// 用户继承,处理IMediaPlayer对应回调逻辑
class IMediaPlayerObserver {
   public:
    virtual ~IMediaPlayerObserver(){};

   public:
    // prepared后,能拿到MediaPlayer的视频与音频信息
    virtual void onPrepared(){};
    virtual void onError(PlayStatus staus, int32_t code, const char* msg){};
    virtual void onVideoFrame(const VideoFrame& frame){};
    virtual void onAudioFrame(const AudioFrame& frame){};
    virtual void onStop(){};
    virtual void onComplate(){};
};

class IMediaPlayer {
   public:
    virtual ~IMediaPlayer(){};
    virtual void setObserver(IMediaPlayerObserver* observer) = 0;
    // 文件路径,URL(RTMP这些)
    virtual void setDataSource(const char* path) = 0;
    virtual void prepare(bool bAsync) = 0;
    // 需要同步的prepare需要之后才能拿到,异步在回调里的onPrepared
    virtual const AudioStream& getAudioStream() = 0;
    virtual const VideoStream& getVideoStream() = 0;
    // 同步prepare可以在下面直接调用start,否则需要在observer里的prepare调用
    virtual void start() = 0;
    virtual void pause() = 0;
    virtual void stop() = 0;
    virtual void release() = 0;
};

class MediaFactory {
   public:
    MediaFactory(){};
    virtual ~MediaFactory(){};

   public:
    virtual IMediaPlayer* createPlay() = 0;
};

}  // namespace aoce