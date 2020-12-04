#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "IModule.hpp"
namespace aoce {

typedef IModule* (*loadModuleAction)();
typedef std::function<IModule*(void)> loadModuleHandle;

class ModuleInfo {
   public:
    void* handle = nullptr;
    IModule* module = nullptr;
    std::string name = "";
    loadModuleHandle onLoadEvent = nullptr;
    bool load = false;

   public:
    ModuleInfo(/* args */){};
    ~ModuleInfo(){};
};

class ACOE_EXPORT ModuleManager {
   public:
    static ModuleManager& Get();
    ~ModuleManager();

   protected:
    ModuleManager();

   private:
    static ModuleManager* instance;
    std::map<std::string, ModuleInfo*> modules;

   private:
    ModuleManager(const ModuleManager&) = delete;
    ModuleManager& operator=(const ModuleManager&) = delete;
    /* data */
   public:
    void registerModule(const char* name,
                        loadModuleHandle handle = nullptr);
    void loadModule(const char* name);
    void unloadModule(const char* name);
    void regAndLoad(const char* name);
};

template <class module>
class StaticLinkModule {
   public:
    StaticLinkModule(const std::string& name) {
        auto& fun = std::bind(&StaticLinkModule<module>::InitModeule, this);
        ModuleManager::Get().registerModule(name, fun);
    }

   public:
    IModule* InitModeule() { return new module(); }
};
}  // namespace aoce
