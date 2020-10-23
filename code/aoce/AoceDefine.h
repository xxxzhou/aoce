#pragma once
#include <functional>

#ifdef _MSC_VER
#if defined(AOCE_EXPORT_DEFINE)
#define ACOE_EXPORT __declspec(dllexport)
#else
#define ACOE_EXPORT __declspec(dllimport)
#endif
#else
#define ACOE_EXPORT
#endif

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#elif __ANDROID__
#define DLLEXPORT __attribute__((visibility("default")))
#else
#define DLLEXPORT
#endif

#if defined(__ANDROID__)

#include <android/asset_manager.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <sys/system_properties.h>

// Missing from the NDK
// namespace std {
// template <typename T, typename... Args>
// std::unique_ptr<T> make_unique(Args&&... args) {
//    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
//}
//}  // namespace std

#define LOGI(...) \
    ((void)__android_log_print(ANDROID_LOG_INFO, "aoce", __VA_ARGS__))
#define LOGW(...) \
    ((void)__android_log_print(ANDROID_LOG_WARN, "aoce", __VA_ARGS__))
#define LOGE(...) \
    ((void)__android_log_print(ANDROID_LOG_ERROR, "aoce", __VA_ARGS__))
#define LOGD(...) \
    ((void)__android_log_print(ANDROID_LOG_DEBUG, "aoce", __VA_ARGS__))

#endif

// 后期会考虑使用静态链接
#define AOCE_STATICLINK 0

#if AOCE_STATICLINK
#define ADD_MODULE(ModuleClass, ModuleName)                            \
    static aoce::StaticLinkModule<ModuleClass> LinkModule##ModuleName( \
        #ModuleName);
#else
#define ADD_MODULE(ModuleClass, ModuleName) \
    extern "C" DLLEXPORT IModule* NewModule() { return new ModuleClass(); }
#endif

#define NOMINMAX 1

// #ifndef max
// #define max(a, b) (((a) > (b)) ? (a) : (b))
// #endif

// #ifndef min
// #define min(a, b) (((a) < (b)) ? (a) : (b))
// #endif

//回调定义,C++传值改用std::function.
// typedef C++ function后缀定义 handle,C为action,前缀不要加on
// 定义的handle变量为on前缀,event后缀
// 定义的handle包装方法为on前缀,handle后缀
// 定义的set handle变量的方法为set前缀(不包含on),handle后缀

//日志回调
typedef void (*logEventAction)(int32_t level, const char* message);
typedef std::function<void(int32_t level, const char* message)> logEventHandle;

typedef void (*imageProcessAction)(uint8_t* data, int32_t width, int32_t height,
                                   int32_t outIndex);
typedef std::function<void(uint8_t* data, int32_t width, int32_t height,
                           int32_t outIndex)>
    imageProcessHandle;
