#include "LiveRoom.hpp"

#include <algorithm>

#define CHECK_AOCE_LIVE_INIT                                       \
    if (roomType == RoomType::noInit) {                            \
        logMessage(LogLevel::error, "live module is init failed"); \
        return false;                                              \
    }

// 检查是否在房间里
#define CHEACK_AOCE_LIVE_INROOM(RETURN_VALUE)                    \
    if (roomType != RoomType::login) {                           \
        logMessage(LogLevel::error, "live module is not login"); \
        return RETURN_VALUE;                                     \
    }

namespace aoce {

LiveRoom::LiveRoom(/* args */) {}

LiveRoom::~LiveRoom() {}

int32_t LiveRoom::getPullIndex(int32_t userId, int32_t index) {
    auto size = pullStreams.size();
    for (int i = 0; i < size; i++) {
        if (pullStreams[i].userId == userId &&
            pullStreams[i].streamId == index) {
            return i;
        }
    }
    return -1;
}

void LiveRoom::resetStreams() {
    pushStreams.clear();
    pullStreams.clear();
}

bool LiveRoom::initRoom(void* liveContext, LiveCallback* liveBack) {
    if (roomType != RoomType::noInit) {
        logMessage(LogLevel::info, "live module is have inited");
        return true;
    }
    bool bInit = onInitRoom(liveContext);
    if (bInit) {
        roomType = RoomType::init;
        this->liveBack = liveBack;
        assert(liveBack != nullptr);
        this->liveBack->onInitRoom();
    } else {
        logMessage(LogLevel::error, "live module is init failed");
    }
    return bInit;
}

bool LiveRoom::loginRoom(const std::string& roomName, int32_t useId,
                         int32_t pushCount) {
    CHECK_AOCE_LIVE_INIT
    if (roomType == RoomType::logining) {
        logMessage(LogLevel::warn, "live is logining.");
        return false;
    } else if (roomType == RoomType::login) {
        logMessage(LogLevel::warn, "live is login.");
        return true;
    }
    this->roomName = roomName;
    this->userId = useId;
    this->pushCount = pushCount;
    if (pushCount > 0) {
        pushStreams.resize(pushCount);
    }
    bool blogining = onLoginRoom();
    if (blogining) {
        // 请注意,在子类里确认登陆后所状态修改成login
        roomType = RoomType::logining;
    }
    return blogining;
}

bool LiveRoom::pushStream(int32_t index, const PushSetting& setting) {
    CHEACK_AOCE_LIVE_INROOM(false)
    if (bDyncPush) {
        if (index >= pushStreams.size()) {
            pushStreams.resize(index + 1);
        }
    } else {
        assert(index < pushStreams.size());
    }
    if (pushStreams[index].bOpen) {
        std::string msg;
        string_format(msg, "push stream index: ", index, " is opened.");
        logMessage(LogLevel::warn, msg);
        return false;
    }
    pushStreams[index].setting = setting;
    bool bopen = onPushStream(index);
    // 请注意,这里可能出现 本地推流操作正确,但是服务器应答出错,
    // 例如liveBack应答的onStreamUpdate不正确,请继续把bopen置为false
    pushStreams[index].bOpen = bopen;
    return bopen;
}
void LiveRoom::stopPushStream(int32_t index) {
    CHEACK_AOCE_LIVE_INROOM(void())
    assert(index < pushStreams.size());
    pushStreams[index].bOpen = false;
    // 需要保证不会推流了,不需要保证正常关闭流,如果关闭流不正常,返回false
    onStopPushStream(index);
}

bool LiveRoom::pushVideoFrame(int32_t index, const VideoFrame& videoFrame) {
    CHEACK_AOCE_LIVE_INROOM(false)
    assert(index < pushStreams.size());
    return onPushVideoFrame(index, videoFrame);
}
bool LiveRoom::pushAudioFrame(int32_t index, const AudioFrame& audioFrame) {
    CHEACK_AOCE_LIVE_INROOM(false)
    assert(index < pushStreams.size());
    return onPushAudioFrame(index, audioFrame);
}

bool LiveRoom::pullStream(int32_t userId, int32_t index,
                          const PullSetting& setting) {
    CHEACK_AOCE_LIVE_INROOM(false)
    int32_t pullIndex = getPullIndex(userId, index);
    if (pullIndex < 0) {
        PullStream stream = {};
        stream.userId = userId;
        stream.streamId = index;
        // stream.bOpen = false;
        // stream.setting = setting;
        pullStreams.push_back(stream);
        pullIndex = pullStreams.size() - 1;
    }
    if (pullStreams[pullIndex].bOpen) {
        std::string msg;
        string_format(msg, "pull stream is opened. user: ", userId,
                      " index: ", index);
        logMessage(LogLevel::warn, msg);
        return false;
    }
    pullStreams[pullIndex].setting = setting;
    // 同上面一样,这只能表明本地是否成功,最终判定正确还需要服务器应答正确
    bool bopen = onPullStream(userId, index);
    pullStreams[pullIndex].bOpen = bopen;
    return bopen;
}

void LiveRoom::stopPullStream(int32_t userId, int32_t index) {
    CHEACK_AOCE_LIVE_INROOM(void())
    int32_t pullIndex = getPullIndex(userId, index);
    assert(pullIndex >= 0);
    pullStreams[pullIndex].bOpen = false;
    onStopPullStream(userId, index);
}

// 退出房间,对应loginRoom
void LiveRoom::logoutRoom() {
    CHEACK_AOCE_LIVE_INROOM(void())
    // 关闭拉流
    for (int32_t i = 0; i < pullStreams.size(); i++) {
        if (pullStreams[i].bOpen) {
            stopPullStream(pullStreams[i].userId, pullStreams[i].streamId);
        }
    }
    for (int32_t i = 0; i < pushStreams.size(); i++) {
        if (pushStreams[i].bOpen) {
            stopPushStream(i);
        }
    }
    onLogoutRoom();
    roomType = RoomType::logout;
}

// 关闭当前房间,对应initRoom
void LiveRoom::shutdownRoom() {
    if (roomType == RoomType::login) {
        logoutRoom();
    }
    // 子类清理资源
    onShutdownRoom();
    roomType = RoomType::noInit;
}

}  // namespace aoce
