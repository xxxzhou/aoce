#pragma once

#include "aoce/module/IModule.hpp"

namespace aoce {
namespace win {

class WinModule : public IModule {
   private:
    /* data */
   public:
    WinModule(/* args */);
    ~WinModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace win
}  // namespace aoce