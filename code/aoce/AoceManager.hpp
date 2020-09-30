#pragma once
#include <map>

#include "Aoce.hpp"
#include "Layer/LayerFactory.hpp"
#include "Layer/PipeGraph.hpp"
#include "VideoDevice/VideoManager.hpp"
namespace aoce {

#define AOCE_MANAGER_OBJ(OBJTYPE, OBJCLASS)                        \
    typedef std::shared_ptr<OBJCLASS> OBJCLASS##Ptr;               \
                                                                   \
   private:                                                        \
    std::map<OBJTYPE, OBJCLASS##Ptr> OBJCLASS##Map;                \
                                                                   \
   public:                                                         \
    inline void add##OBJCLASS(OBJTYPE s_type, OBJCLASS* manager) { \
        OBJCLASS##Ptr ptr(manager);                                \
        OBJCLASS##Map[s_type] = ptr;                               \
    }                                                              \
    inline void remove##OBJCLASS(OBJTYPE s_type) {                 \
        OBJCLASS##Map[s_type] = nullptr;                           \
    }                                                              \
    inline OBJCLASS##Ptr get##OBJCLASS(OBJTYPE s_type) {           \
        return OBJCLASS##Map[s_type];                              \
    }

class ACOE_EXPORT AoceManager {
   public:
    static AoceManager& Get();
    // 清理资源
    static void clean();

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