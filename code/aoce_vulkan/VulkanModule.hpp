#pragma once

#include <Module/IModule.hpp>

namespace aoce {
namespace win {
namespace mf {
class VulkanModule : public IModule {
   private:
    /* data */
   public:
    VulkanModule(/* args */);
    ~VulkanModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};
}  // namespace mf
}  // namespace win
}  // namespace aoce