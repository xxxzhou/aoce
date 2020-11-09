#pragma once

#include <Module/IModule.hpp>

namespace aoce {
class FFmpegModule : public IModule {
   private:
    /* data */
   public:
    FFmpegModule(/* args */);
    virtual ~FFmpegModule() override;

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace aoce