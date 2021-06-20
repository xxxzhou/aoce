#pragma once

#include "aoce/module/IModule.hpp"

namespace aoce {
namespace awinrt {

class WinRTModule : public IModule {
   private:
    /* data */
   public:
    WinRTModule(/* args */);
    ~WinRTModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace win
}  // namespace aoce