#pragma once

#include <media/MediaPlayer.hpp>

namespace aoce {
namespace ffmpeg {

class FMediaFactory : public MediaFactory {
   private:
    /* data */
   public:
    FMediaFactory(/* args */);
    virtual ~FMediaFactory() override;

   public:
    virtual IMediaPlayer* createPlay() override;
    virtual IMediaMuxer* createMuxer() override;
};

}  // namespace ffmpeg
}  // namespace aoce