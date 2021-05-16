#pragma once

#include <module/IModule.hpp>

namespace aoce {
namespace android {

class AndroidModule : public IModule {
   private:
    /* data */
   public:
    AndroidModule(/* args */);
    ~AndroidModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace android
}  // namespace aoce