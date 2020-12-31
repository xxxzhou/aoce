#pragma once

#include <Module/IModule.hpp>

namespace aoce {
namespace cuda {
class CudaModule : public IModule {
   private:
    /* data */
   public:
    CudaModule(/* args */);
    ~CudaModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};
}  // namespace cuda
}  // namespace aoce