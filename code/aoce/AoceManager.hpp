#pragma once
#include <map>

#include "Aoce.hpp"
#include "Layer/LayerFactory.hpp"
#include "Layer/PipeGraph.hpp"
#include "VideoDevice/VideoManager.hpp"
struct android_app;
namespace aoce {

    // struct android_app;

#define AOCE_MANAGER_OBJ(OBJTYPE, OBJCLASS)                        \
    typedef std::unique_ptr<OBJCLASS> OBJCLASS##Ptr;               \
                                                                   \
   private:                                                        \
    std::map<OBJTYPE, OBJCLASS##Ptr> OBJCLASS##Map;                \
                                                                   \
   public:                                                         \
    inline void add##OBJCLASS(OBJTYPE s_type, OBJCLASS* manager) { \
        OBJCLASS##Map[s_type] = OBJCLASS##Ptr(manager);            \
    }                                                              \
    inline void remove##OBJCLASS(OBJTYPE s_type) {                 \
        OBJCLASS##Map[s_type].reset();                             \
    }                                                              \
    inline OBJCLASS* get##OBJCLASS(OBJTYPE s_type) {               \
        return OBJCLASS##Map[s_type].get();                        \
    }

class ACOE_EXPORT AoceManager {
   public:
    static AoceManager& Get();
    // 清理资源
    static void clean();

   private:
#if __ANDROID__
    android_app* androidApp = nullptr;
#endif
   public:
#if __ANDROID__
    inline void initAndroid(android_app* app) { androidApp = app; }
    inline android_app* getApp() { return androidApp; }
#endif
   private:
    AoceManager(/* args */);
    static AoceManager* instance;
    AoceManager(const AoceManager&) = delete;
    AoceManager& operator=(const AoceManager&) = delete;

   public:
    ~AoceManager();

    AOCE_MANAGER_OBJ(CameraType, VideoManager)
    AOCE_MANAGER_OBJ(GpuType, PipeGraphFactory)
    AOCE_MANAGER_OBJ(GpuType, LayerFactory)
};

}  // namespace aoce