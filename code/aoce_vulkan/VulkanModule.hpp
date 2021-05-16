#pragma once

#include <module/IModule.hpp>

namespace aoce {
namespace vulkan {
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
}  // namespace vulkan
}  // namespace aoce