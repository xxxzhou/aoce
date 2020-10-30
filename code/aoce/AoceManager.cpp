#include "AoceManager.hpp"

namespace aoce {

    AoceManager *AoceManager::instance = nullptr;

    AoceManager &AoceManager::Get() {
        if (instance == nullptr) {
            instance = new AoceManager();
        }
        return *instance;
    }

    AoceManager::AoceManager(/* args */) {}

    AoceManager::~AoceManager() {}

#if __ANDROID__

    void *AoceManager::getJNIContext() {
        jclass activityThreadCls = env->FindClass("android/app/ActivityThread");
        jmethodID currentActivityThread = env->GetStaticMethodID(activityThreadCls,
                                                                 "currentActivityThread",
                                                                 "()Landroid/app/ActivityThread;");
        jobject activityThreadObj =
                env->CallStaticObjectMethod(activityThreadCls, currentActivityThread);

        jmethodID getApplication =
                env->GetMethodID(activityThreadCls, "getApplication",
                                 "()Landroid/app/Application;");
        jobject context = env->CallObjectMethod(activityThreadObj, getApplication);
        return context;
    }

#endif
}  // namespace aoce