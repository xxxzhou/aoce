#pragma once

#include <AoceCore.h>

namespace aoce {
namespace win {
namespace mf {

class MfModule : public IModule {
   private:
    /* data */
   public:
    MfModule(/* args */);
    ~MfModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace mf

}  // namespace win
}  // namespace aoce