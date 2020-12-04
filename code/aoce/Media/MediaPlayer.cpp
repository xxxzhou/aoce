#include "MediaPlayer.hpp"

#include <map>

namespace aoce {

const std::map<MediaType, std::vector<std::string>> MediaTypeMap = {
    {MediaType::rtmp, {"rtmp://"}},
    {MediaType::http, {"http://", "udp://"}},
    {MediaType::file, {".mp4", ".rmvb"}},
};

MediaType getMediaType(const std::string& str) {
    for (const auto& kv : MediaTypeMap) {
        for (const auto& suffix : kv.second) {
            // 后缀
            if (suffix.find(".") == 0) {
                if (str.find_last_of(suffix) == str.size() - 1) {
                    return kv.first;
                }
            } else {
                if (str.find(suffix) == 0) {
                    return kv.first;
                }
            }
        }
    }
    return MediaType::other;
}

void MediaPlayer::setObserver(IMediaPlayerObserver* observer) {
    this->observer = observer;
    onSetObserver();
}

void MediaPlayer::setDataSource(const char* path) {
    logAssert(status == PlayStatus::idle,
              "media play current status is not idle");
    uri = path;
    mediaType = getMediaType(uri);
    onSetDataSource();
    status = PlayStatus::init;
}

}  // namespace aoce