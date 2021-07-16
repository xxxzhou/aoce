#include "MediaPlayer.hpp"

#include <map>

namespace aoce {

void MediaPlayer::setObserver(IMediaPlayerObserver* observer) {
    this->observer = observer;
    onSetObserver();
}

void MediaPlayer::setDataSource(const char* path) {
    if (status != PlayStatus::idle) {
        logMessage(LogLevel::error, "media play current status is not idle");
    }
    uri = path;
    mediaType = getMediaType(uri);
    onSetDataSource();
    status = PlayStatus::init;
}

}  // namespace aoce