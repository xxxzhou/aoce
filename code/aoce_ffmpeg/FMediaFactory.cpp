#include "FMediaFactory.hpp"

#include "media/FMediaMuxer.hpp"
#include "media/FMediaPlayer.hpp"

namespace aoce {
namespace ffmpeg {

FMediaFactory::FMediaFactory(/* args */) {}

FMediaFactory::~FMediaFactory() {}

IMediaPlayer* FMediaFactory::createPlay() { return new FMediaPlayer(); }

IMediaMuxer* FMediaFactory::createMuxer() { return new FMediaMuxer(); }

}  // namespace ffmpeg
}  // namespace aoce