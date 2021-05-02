#pragma once
#include <functional>

#include "AoceBuildSettings.h"

#ifdef _WIN32
#if defined(AOCE_EXPORT_DEFINE)
#define ACOE_EXPORT __declspec(dllexport)
#else
#define ACOE_EXPORT __declspec(dllimport)
#endif
#elif __ANDROID__
#if defined(AOCE_EXPORT_DEFINE)
#define ACOE_EXPORT __attribute__((visibility("default")))
#else
#define ACOE_EXPORT
#endif
#endif

#ifdef _WIN32
#define AOCE_DLL_EXPORT __declspec(dllexport)
#elif __ANDROID__
#define AOCE_DLL_EXPORT __attribute__((visibility("default")))
#else
#define AOCE_DLL_EXPORT
#endif

#if __ANDROID__

#include <android/asset_manager.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <sys/system_properties.h>

//// Missing from the NDK
// namespace std {
// template <typename T, typename... Args>
// std::unique_ptr<T> make_unique(Args&&... args) {
//   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
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

#if AOCE_USE_STATIC
#define ADD_MODULE(ModuleClass, ModuleName)                            \
    static aoce::StaticLinkModule<ModuleClass> LinkModule##ModuleName( \
        #ModuleName);
#else
#define ADD_MODULE(ModuleClass, ModuleName)           \
    extern "C" AOCE_DLL_EXPORT IModule* NewModule() { \
        return new ModuleClass();                     \
    }
#endif

#define NOMINMAX 1

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif

// 采用opencv里的数据描述,前三BIT表示类型，后三BIT表示通道个数
#pragma region opencv
#define AOCE_CV_CN_MAX 512
#define AOCE_CV_CN_SHIFT 3
#define AOCE_CV_DEPTH_MAX (1 << AOCE_CV_CN_SHIFT)

#define AOCE_CV_8U 0
#define AOCE_CV_8S 1
#define AOCE_CV_16U 2
#define AOCE_CV_16S 3
#define AOCE_CV_32S 4
#define AOCE_CV_32F 5
#define AOCE_CV_64F 6
#define AOCE_CV_16F 7

#define AOCE_CV_MAT_DEPTH_MASK (AOCE_CV_DEPTH_MAX - 1)
#define AOCE_CV_MAT_DEPTH(flags) ((flags)&AOCE_CV_MAT_DEPTH_MASK)

#define AOCE_CV_MAKETYPE(depth, cn) \
    (AOCE_CV_MAT_DEPTH(depth) + (((cn)-1) << AOCE_CV_CN_SHIFT))
#define AOCE_CV_MAKE_TYPE AOCE_CV_MAKETYPE

#define AOCE_CV_8UC1 AOCE_CV_MAKETYPE(AOCE_CV_8U, 1)
#define AOCE_CV_8UC2 AOCE_CV_MAKETYPE(AOCE_CV_8U, 2)
#define AOCE_CV_8UC3 AOCE_CV_MAKETYPE(AOCE_CV_8U, 3)
#define AOCE_CV_8UC4 AOCE_CV_MAKETYPE(AOCE_CV_8U, 4)
#define AOCE_CV_8UC(n) AOCE_CV_MAKETYPE(AOCE_CV_8U, (n))

#define AOCE_CV_8SC1 AOCE_CV_MAKETYPE(AOCE_CV_8S, 1)
#define AOCE_CV_8SC2 AOCE_CV_MAKETYPE(AOCE_CV_8S, 2)
#define AOCE_CV_8SC3 AOCE_CV_MAKETYPE(AOCE_CV_8S, 3)
#define AOCE_CV_8SC4 AOCE_CV_MAKETYPE(AOCE_CV_8S, 4)
#define AOCE_CV_8SC(n) AOCE_CV_MAKETYPE(AOCE_CV_8S, (n))

#define AOCE_CV_16UC1 AOCE_CV_MAKETYPE(AOCE_CV_16U, 1)
#define AOCE_CV_16UC2 AOCE_CV_MAKETYPE(AOCE_CV_16U, 2)
#define AOCE_CV_16UC3 AOCE_CV_MAKETYPE(AOCE_CV_16U, 3)
#define AOCE_CV_16UC4 AOCE_CV_MAKETYPE(AOCE_CV_16U, 4)
#define AOCE_CV_16UC(n) AOCE_CV_MAKETYPE(AOCE_CV_16U, (n))

#define AOCE_CV_16SC1 AOCE_CV_MAKETYPE(AOCE_CV_16S, 1)
#define AOCE_CV_16SC2 AOCE_CV_MAKETYPE(AOCE_CV_16S, 2)
#define AOCE_CV_16SC3 AOCE_CV_MAKETYPE(AOCE_CV_16S, 3)
#define AOCE_CV_16SC4 AOCE_CV_MAKETYPE(AOCE_CV_16S, 4)
#define AOCE_CV_16SC(n) AOCE_CV_MAKETYPE(AOCE_CV_16S, (n))

#define AOCE_CV_32SC1 AOCE_CV_MAKETYPE(AOCE_CV_32S, 1)
#define AOCE_CV_32SC2 AOCE_CV_MAKETYPE(AOCE_CV_32S, 2)
#define AOCE_CV_32SC3 AOCE_CV_MAKETYPE(AOCE_CV_32S, 3)
#define AOCE_CV_32SC4 AOCE_CV_MAKETYPE(AOCE_CV_32S, 4)
#define AOCE_CV_32SC(n) AOCE_CV_MAKETYPE(AOCE_CV_32S, (n))

#define AOCE_CV_32FC1 AOCE_CV_MAKETYPE(AOCE_CV_32F, 1)
#define AOCE_CV_32FC2 AOCE_CV_MAKETYPE(AOCE_CV_32F, 2)
#define AOCE_CV_32FC3 AOCE_CV_MAKETYPE(AOCE_CV_32F, 3)
#define AOCE_CV_32FC4 AOCE_CV_MAKETYPE(AOCE_CV_32F, 4)
#define AOCE_CV_32FC(n) AOCE_CV_MAKETYPE(AOCE_CV_32F, (n))

#define AOCE_CV_64FC1 AOCE_CV_MAKETYPE(AOCE_CV_64F, 1)
#define AOCE_CV_64FC2 AOCE_CV_MAKETYPE(AOCE_CV_64F, 2)
#define AOCE_CV_64FC3 AOCE_CV_MAKETYPE(AOCE_CV_64F, 3)
#define AOCE_CV_64FC4 AOCE_CV_MAKETYPE(AOCE_CV_64F, 4)
#define AOCE_CV_64FC(n) AOCE_CV_MAKETYPE(AOCE_CV_64F, (n))

#define AOCE_CV_16FC1 AOCE_CV_MAKETYPE(AOCE_CV_16F, 1)
#define AOCE_CV_16FC2 AOCE_CV_MAKETYPE(AOCE_CV_16F, 2)
#define AOCE_CV_16FC3 AOCE_CV_MAKETYPE(AOCE_CV_16F, 3)
#define AOCE_CV_16FC4 AOCE_CV_MAKETYPE(AOCE_CV_16F, 4)
#define AOCE_CV_16FC(n) AOCE_CV_MAKETYPE(AOCE_CV_16F, (n))

#define AOCE_CV_MAT_CN_MASK ((AOCE_CV_CN_MAX - 1) << AOCE_CV_CN_SHIFT)
#define AOCE_CV_MAT_CN(flags) \
    ((((flags)&AOCE_CV_MAT_CN_MASK) >> AOCE_CV_CN_SHIFT) + 1)
/** Size of each channel item,
   0x28442211 = 0010 1000 0100 0100 0010 0010 0001 0001 ~ array of
   sizeof(arr_type_elem) */
#define AOCE_CV_ELEM_SIZE1(type) \
    ((0x28442211 >> AOCE_CV_MAT_DEPTH(type) * 4) & 15)
#define AOCE_CV_ELEM_SIZE(type) \
    (AOCE_CV_MAT_CN(type) * AOCE_CV_ELEM_SIZE1(type))
#pragma endregion

// #ifndef max
// #define max(a, b) (((a) > (b)) ? (a) : (b))
// #endif

// #ifndef min
// #define min(a, b) (((a) < (b)) ? (a) : (b))
// #endif

// 回调定义,C++传值改用std::function.
// typedef C++ function后缀定义 handle,C为action,前缀不要加on
// 定义的handle变量为on前缀,event后缀
// 定义的handle包装方法为on前缀,handle后缀
// 定义的set handle变量的方法为set前缀(不包含on),handle后缀

//日志回调
typedef void (*logEventAction)(int32_t level, const char* message);
typedef std::function<void(int32_t level, const char* message)> logEventHandle;
