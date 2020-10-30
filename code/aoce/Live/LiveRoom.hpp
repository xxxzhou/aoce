#pragma once

#include "../Aoce.hpp"
#include "LiveCallback.hpp"
namespace aoce {

struct PushSetting {
    // 是否拉取声音
    int32_t bAudio = 1;
    // 是否拉取视频
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
    // 是否自动播放声音(如果为0,则自己去处理声音)
    int32_t bPlayAudio = 1;
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

enum class RoomType {
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
};

class ACOE_EXPORT LiveRoom {
   protected:
    // 推流设置
    std::vector<PushStream> pushStreams;
    // 拉流设置
    std::vector<PullStream> pullStreams;
    std::string roomName = "";
    int32_t userId = -1;
    // 子类一定登陆后一定要修改成login状态,只有子类才知道确认时机
    RoomType roomType = RoomType::noInit;
    LiveCallback* liveBack = nullptr;
    // 默认推一个流
    int32_t pushCount = 1;
    // 如果为true,可以动态添加推流,则不需要在登录时指定推流个数
    bool bDyncPush = false;

   public:
    LiveRoom(/* args */);
    virtual ~LiveRoom();

    // 回调对应子类具体的公共方法实现
   protected:
    // 子类initRoom的实现
    virtual bool onInitRoom(void* liveContext) = 0;
    virtual bool onLoginRoom() = 0;
    virtual bool onPushStream(int32_t index) = 0;
    virtual void onStopPushStream(int32_t index) = 0;
    virtual bool onPushVideoFrame(int32_t index, const VideoFrame& videoFrame) {
        return false;
    };
    virtual bool onPushAudioFrame(int32_t index, const AudioFrame& audioFrame) {
        return false;
    };
    virtual bool onPullStream(int32_t userId, int32_t index) = 0;
    virtual void onStopPullStream(int32_t userId, int32_t index) = 0;
    virtual bool onLogoutRoom() = 0;
    virtual void onShutdownRoom() = 0;

   public:
    int32_t getPullIndex(int32_t userId, int32_t index);
    void resetStreams();

   public:
    // 因为每个直播SDK初始化信息不相同,简单使用void*表示
    bool initRoom(void* liveContext, LiveCallback* liveBack);
    // 登陆房间,房间名,用户id,推流个数(这个会影响一些设置)
    bool loginRoom(const std::string& roomName, int32_t useId,
                   int32_t pushCount);

    bool pushStream(int32_t index, const PushSetting& setting);
    void stopPushStream(int32_t index);

    bool pushVideoFrame(int32_t index, const VideoFrame& videoFrame);
    bool pushAudioFrame(int32_t index, const AudioFrame& audioFrame);

    bool pullStream(int32_t userId, int32_t index, const PullSetting& setting);
    void stopPullStream(int32_t userId, int32_t index);

    // 退出房间,对应loginRoom
    void logoutRoom();
    // 关闭当前房间,对应initRoom
    void shutdownRoom();
};

}  // namespace aoce