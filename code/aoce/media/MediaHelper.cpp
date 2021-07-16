#include "MediaHelper.hpp"

#include <map>

namespace aoce {

const std::map<MediaSourceType, std::vector<std::string>> MediaTypeMap = {
    {MediaSourceType::rtmp, {"rtmp://"}},
    {MediaSourceType::http, {"http://", "udp://"}},
    {MediaSourceType::file, {".mp4", ".rmvb"}},
};

MediaSourceType getMediaType(const std::string& str) {
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
    return MediaSourceType::other;
}

std::string getAvformat(const std::string& uri) {
    if (uri.find("rtmp://") == 0) {
        return "flv";
    } else if (uri.find("http://") == 0 || uri.find("udp://") == 0) {
        return "mpegts";
    } else if (uri.find("rtsp://") == 0) {
        return "rtsp";
    } else if (uri.find(".mp4") != std::string::npos) {
        return "mp4";
    }
    return "flv";
}

}  // namespace aoce