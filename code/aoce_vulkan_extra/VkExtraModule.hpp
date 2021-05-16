#pragma once

#include <module/IModule.hpp>

namespace aoce {
namespace vulkan {
    
class VkExtraModule : public IModule {
   private:
    /* data */
   public:
    VkExtraModule(/* args */);
    ~VkExtraModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace vulkan
}  // namespace aoce