#pragma once

#include "../Aoce.hpp"
namespace aoce {

class ACOE_EXPORT IModule {
   private:
    /* data */
    void* handle;

   public:
    IModule(/* args */);
    virtual ~IModule();

   public:
    // load module时调用
    virtual bool loadModule() { return false; };
    // unload module时调用
    virtual void unloadModule(){};
};

}  // namespace aoce
