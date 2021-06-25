#pragma once

#include "../Aoce.hpp"

namespace aoce {

class ACOE_EXPORT LiveRoom : public ILiveRoom {
   protected:
    // 推流设置
    std::vector<PushStream> pushStreams;
    // 拉流设置
    std::vector<PullStream> pullStreams;
    std::string roomName = "";
    int32_t userId = -1;
    // 子类一定登陆后一定要修改成login状态,只有子类才知道确认时机
    RoomType roomType = RoomType::noInit;
    ILiveObserver* liveBack = nullptr;
    // 默认推一个流
    int32_t pushCount = 1;
    // 如果为true,可以动态添加推流,则不需要在登录时指定推流个数
    bool bDyncPush = false;
    float micVolume = 0.0f;

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

   protected:
    void resetStreams();

   public:
    virtual int32_t getUserId() final;
    virtual int32_t getPullIndex(int32_t userId, int32_t index) final;
    virtual float getMicVolume() final;

   public:
    // 设置播放拉流音量,0静音,100最大,如果自己处理音量数据,请设置为0
    virtual void setPlayVolume(int32_t value) override{};

   public:
    // 因为每个直播SDK初始化信息不相同,简单使用void*表示
    virtual bool initRoom(void* liveContext, ILiveObserver* liveBack) final;
    // 登陆房间,房间名,用户id,推流个数(这个会影响一些设置)
    virtual bool loginRoom(const char* roomName, int32_t useId,
                           int32_t pushCount) final;

    virtual bool pushStream(int32_t index, const PushSetting& setting) final;
    virtual void stopPushStream(int32_t index) final;

    virtual bool pushVideoFrame(int32_t index,
                                const VideoFrame& videoFrame) final;
    virtual bool pushAudioFrame(int32_t index,
                                const AudioFrame& audioFrame) final;

    virtual bool pullStream(int32_t userId, int32_t index,
                            const PullSetting& setting) final;
    virtual void stopPullStream(int32_t userId, int32_t index) final;

    // 退出房间,对应loginRoom
    virtual void logoutRoom() final;
    // 关闭当前房间,对应initRoom
    virtual void shutdownRoom() final;
};

}  // namespace aoce