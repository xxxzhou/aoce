#pragma once

#include "../Aoce.hpp"

namespace aoce {

// 请用户继承此类,相应回调会转到此类上
// 网络交互,所以相应流程 提交->服务器应答->处理
// 这里的回调是相应提交后返回的服务器应答,在子类应答里填写的实现就是处理
class ACOE_EXPORT ILiveObserver {
   private:
    /* data */
   public:
    ILiveObserver(/* args */);
    virtual ~ILiveObserver();

   public:
    // 网络发生的各种情况与处理码,如断网,网络情况不好等
    virtual void onEvent(int32_t operater, int32_t code, LogLevel level,
                         const std::string& msg){};
    // 
    virtual void onInitRoom(){};
    // loginRoom的网络应答
    virtual void onLoginRoom(bool bReConnect = false){};
    // 加入的房间人数变化
    virtual void onUserChange(int32_t userId, bool bAdd) = 0;
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
    // 推流的质量
    virtual void onPushQuality(int32_t index, int32_t quality, float fps,
                               float kbs){};
    // 拉流质量
    virtual void onPullQuality(int32_t userId, int32_t index, int32_t quality,
                               float fps, float kbs){};
    // 拿出房间
    virtual void onLogoutRoom(){};
};

}  // namespace aoce