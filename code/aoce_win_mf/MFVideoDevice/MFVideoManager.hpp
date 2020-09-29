#pragma once
#include "VideoDevice/VideoManager.hpp"
namespace aoce {
namespace win {
namespace mf {
    
class MFVideoManager : public VideoManager {
   private:
    /* data */
   public:
    MFVideoManager(/* args */);
    virtual ~MFVideoManager() override;

   protected:
    virtual void getDevices() override;
};

}  // namespace mf
}  // namespace win
}  // namespace aoce