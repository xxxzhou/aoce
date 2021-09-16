#pragma once

#include <module/IModule.hpp>

namespace aoce {
 

class VkNcnnModule : public IModule {
   private:
    /* data */
   public:
    VkNcnnModule(/* args */);
    virtual ~VkNcnnModule();

   public:
    virtual bool loadModule() override;
    virtual void unloadModule() override;
};

}  // namespace aoce