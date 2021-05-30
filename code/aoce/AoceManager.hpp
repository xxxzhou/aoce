#pragma once

#include <map>

#include "Aoce.hpp"
#include "layer/PipeGraph.hpp"
#include "live/LiveRoom.hpp"
#include "media/MediaPlayer.hpp"
#include "videodevice/VideoManager.hpp"

#if __ANDROID__
struct android_app;
#endif

namespace aoce {
// struct android_app;
// #if WIN32
#define AOCE_MANAGER_OBJ(OBJTYPE, OBJCLASS)                        \
    typedef std::unique_ptr<OBJCLASS> OBJCLASS##Ptr;               \
                                                                   \
   private:                                                        \
    std::map<OBJTYPE, OBJCLASS##Ptr> OBJCLASS##Map;                \
                                                                   \
   public:                                                         \
    inline void add##OBJCLASS(OBJTYPE s_type, OBJCLASS *manager) { \
        OBJCLASS##Map[s_type] = OBJCLASS##Ptr(manager);            \
    }                                                              \
    inline void remove##OBJCLASS(OBJTYPE s_type) {                 \
        OBJCLASS##Map[s_type].reset();                             \
    }                                                              \
    inline OBJCLASS *get##OBJCLASS(OBJTYPE s_type) {               \
        return OBJCLASS##Map[s_type].get();                        \
    }

class ACOE_EXPORT AoceManager {
   public:
    static AoceManager &Get();
    // 清理资源
    static void clean();

   private:
#if __ANDROID__
    android_app *app = nullptr;
    AndroidEnv androidEnv = {};
    bool bAttach = false;
#endif
   public:
#if __ANDROID__
    // 如果是要用nativewindow,请使用这个
    void initAndroid(android_app *app);
    inline android_app *getApp() { return app; };
    // 如果不是nativewindow,请尽量填充AndroidEnv里的JavaVM等相关
    void initAndroid(const AndroidEnv &andEnv);
    const AndroidEnv &getAppEnv() { return androidEnv; }
    // 要使用jni里的如findcalss/GetStaticMethodID等方法
    // 必需在主线程或是附加主线程里调用
    JNIEnv *getEnv(bool &bAttach);
    // initAndroid里保存了一个env,如果在initAndroid线程里,不需要传入env
    jobject getActivityApplication(jobject activity, JNIEnv *env = nullptr);
    std::string getObjClassName(jobject obj, JNIEnv *env = nullptr);

    // 如果在线程拿过JNIEnv,退出时请调用
    void detachThread();
#endif
   private:
    AoceManager(/* args */);
    static AoceManager *instance;
    AoceManager(const AoceManager &) = delete;
    AoceManager &operator=(const AoceManager &) = delete;

   public:
    ~AoceManager();

   public:
    AOCE_MANAGER_OBJ(CameraType, VideoManager)
    AOCE_MANAGER_OBJ(GpuType, PipeGraphFactory)
    AOCE_MANAGER_OBJ(GpuType, LayerFactory)
    AOCE_MANAGER_OBJ(LiveType, LiveRoom)
    AOCE_MANAGER_OBJ(MediaType, MediaFactory)
};

}  // namespace aoce