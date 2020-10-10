#pragma once

#include <Module/IModule.hpp>

namespace aoce {
namespace vk {
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
}  // namespace vk
}  // namespace aoce