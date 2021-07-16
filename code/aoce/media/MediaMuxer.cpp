#include "MediaMuxer.hpp"

namespace aoce {

MediaMuxer::MediaMuxer(/* args */) {}

MediaMuxer::~MediaMuxer() {}

void MediaMuxer::setOutput(const char* path) {
    url = path;
    mediaType = getMediaType(url);
}

void MediaMuxer::setAudioStream(const AudioStream& stream) {
    audioStream = stream;
    bAudio = true;
}

void MediaMuxer::setVideoStream(const VideoStream& stream) {
    videoStream = stream;
    bVideo = true;
}

}  // namespace aoce