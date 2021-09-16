#pragma once
#include "Aoce.h"

namespace aoce {

struct PushSetting {
    // 是否推声音
    int32_t bAudio = 1;
    // 是否推视频
    int32_t bVideo = 1;
    // 视频信息
    VideoStream videoStream = {};
    // 音频信息
    AudioStream audioStream = {};
    // 是否转推CDN
    int32_t bCDN = false;
};

struct PullSetting {
    // 是否拉取声音
    int32_t bAudio = 1;
    // 是否拉取视频
    int32_t bVideo = 1;
};

struct PullStream {
    int32_t userId = 0;
    int32_t streamId = 0;
    int32_t bOpen = false;
    PullSetting setting = {};
};

struct PushStream {
    int32_t bOpen = false;
    PushSetting setting = {};
};

enum class RoomType : int32_t {
    // 没有初始化
    noInit,
    // 初始化
    init,
    // 登陆房间中
    logining,
    // 登陆成功回调
    login,
    // 登出房间
    logout,
};

struct AgoraContext {
    // 是否软编
    int32_t bSFEncoder = false;
    // 是否自己采集声卡
    int32_t bLoopback = false;
// andrid下需要提供context
#if __ANDROID__
    void* context = nullptr;
#endif
};

// 请用户继承此类,相应回调会转到此类上
// 网络交互,所以相应流程 提交->服务器应答->处理
// 这里的回调是相应提交后返回的服务器应答,在子类应答里填写的实现就是处理
class ILiveObserver {
   public:
    virtual ~ILiveObserver(){};

   public:
    // 网络发生的各种情况与处理码,如断网,网络情况不好等
    virtual void onEvent(int32_t operater, int32_t code, LogLevel level,
                         const char* msg){};
    //
    virtual void onInitRoom(){};
    // loginRoom的网络应答
    virtual void onLoginRoom(bool bReConnect = false){};
    // 加入的房间人数变化
    virtual void onUserChange(int32_t userId, bool bAdd){};
    // 自己pushStream/stopPushStream推流回调,code不为0应该是出错了
    virtual void onStreamUpdate(int32_t index, bool bAdd, int32_t code){};
    // 别的用户pullStream/stopPullStream拉流回调
    virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd,
                                int32_t code){};
    // 用户对应流的视频桢数据
    virtual void onVideoFrame(int32_t userId, int32_t index,
                              const VideoFrame& videoFrame){};
    // 用户对应流的音频桢数据
    virtual void onAudioFrame(int32_t userId, int32_t index,
                              const AudioFrame& audioFrame){};

    // 当前拉流对应用户音量
    virtual void onPlayVolume(int32_t userId, int32_t index, float volume){};

    // 推流的质量
    virtual void onPushQuality(int32_t index, int32_t quality, float fps,
                               float kbs){};
    // 拉流质量
    virtual void onPullQuality(int32_t userId, int32_t index, int32_t quality,
                               float fps, float kbs){};
    // 登出房间
    virtual void onLogoutRoom(){};
};

class ILiveRoom {
   public:
    virtual ~ILiveRoom(){};
    // 得到用户ID
    virtual int32_t getUserId() = 0;
    // 得到拉流索引
    virtual int32_t getPullIndex(int32_t userId, int32_t index) = 0;
    // 得到麦克风音量
    virtual float getMicVolume() = 0;
    // 设置播放拉流音量,0静音,100最大,如果自己处理音量数据,请设置为0
    virtual void setPlayVolume(int32_t value) = 0;
    // 因为每个直播SDK初始化信息不相同,简单使用void*表示
    virtual bool initRoom(void* liveContext, ILiveObserver* liveBack) = 0;
    // 登陆房间,pushCount指明需要推流个数
    virtual bool loginRoom(const char* roomName, int32_t useId,
                           int32_t pushCount) = 0;
    // 指定推流与推流设置
    virtual bool pushStream(int32_t index, const PushSetting& setting) = 0;
    virtual void stopPushStream(int32_t index) = 0;

    virtual bool pushVideoFrame(int32_t index,
                                const VideoFrame& videoFrame) = 0;
    virtual bool pushAudioFrame(int32_t index,
                                const AudioFrame& audioFrame) = 0;
    // 拉流以及拉流设置
    virtual bool pullStream(int32_t userId, int32_t index,
                            const PullSetting& setting) = 0;
    virtual void stopPullStream(int32_t userId, int32_t index) = 0;

    // 退出房间,对应loginRoom
    virtual void logoutRoom() = 0;
    // 关闭当前房间,对应initRoom
    virtual void shutdownRoom() = 0;
};

}  // namespace aoce