#include "AoceManager.hpp"
#if __ANDROID__
#include <android/asset_manager_jni.h>
#include <android/native_activity.h>
#include <android_native_app_glue.h>
#endif
namespace aoce {

AoceManager *AoceManager::instance = nullptr;

AoceManager &AoceManager::Get() {
    if (instance == nullptr) {
        instance = new AoceManager();
    }
    return *instance;
}

AoceManager::AoceManager(/* args */) {
#if __ANDROID__
    // androidEnv = std::make_unique<AndroidEnv>();
#endif
}

AoceManager::~AoceManager() {}

#if __ANDROID__

void AoceManager::initAndroid(android_app *app) {
    this->app = app;
    AndroidEnv temp = {};
    temp.vm = app->activity->vm;
    // temp.env = app->activity->env;
    // temp.activity = app->activity->clazz;
    temp.assetManager = app->activity->assetManager;
    temp.sdkVersion = app->activity->sdkVersion;
    initAndroid(temp);
}

void AoceManager::initAndroid(const AndroidEnv &andEnv) {
    androidEnv = andEnv;
    if (androidEnv.sdkVersion == 0) {
        androidEnv.sdkVersion = JNI_VERSION_1_6;
    }
    jobject active = androidEnv.activity;
    if(!androidEnv.env) {
        JNIEnv *env = getEnv(bAttach);
        androidEnv.env = env;
    }
    JNIEnv *env = androidEnv.env;
    jobject applicationContext = nullptr;
    if (active) {
        jclass contextClass = env->FindClass("android/content/Context");
        jmethodID getApplicationContextMethod =
            env->GetMethodID(contextClass, "getApplicationContext",
                             "()Landroid/content/Context;");
        assert(getApplicationContextMethod != 0);
        applicationContext =
            env->CallObjectMethod(active, getApplicationContextMethod);
    } else {
        //获取Activity Thread的实例对象
        jclass activityThreadCls = env->FindClass("android/app/ActivityThread");
        jmethodID currentActivityThread =
            env->GetStaticMethodID(activityThreadCls, "currentActivityThread",
                                   "()Landroid/app/ActivityThread;");
        jobject activityThread = env->CallStaticObjectMethod(
            activityThreadCls, currentActivityThread);
        //获取Application，也就是全局的Context
        jmethodID getApplication = env->GetMethodID(
            activityThreadCls, "getApplication", "()Landroid/app/Application;");
        applicationContext =
            env->CallObjectMethod(activityThread, getApplication);
    }
    androidEnv.application = env->NewGlobalRef(applicationContext);
    if (androidEnv.application && !androidEnv.assetManager) {
        jmethodID methodGetAssets = env->GetMethodID(
            env->GetObjectClass(androidEnv.application), "getAssets",
            "()Landroid/content/res/AssetManager;");
        jobject localAssetManager =
            env->CallObjectMethod(androidEnv.application, methodGetAssets);
        jobject globalAssetManager = env->NewGlobalRef(localAssetManager);
        androidEnv.assetManager = AAssetManager_fromJava(env, globalAssetManager);
    }
}

// 要使用jni里的如findcalss/GetStaticMethodID,必需附加到主线程里调用
JNIEnv *AoceManager::getEnv(bool &bAttach) {
    JNIEnv *threadEnv = nullptr;
    jint ret =
        androidEnv.vm->GetEnv((void **)&threadEnv, androidEnv.sdkVersion);
    bAttach = false;
    // 没有附加
    // if (ret == JNI_EDETACHED) {
    if (ret < 0) {
        ret = androidEnv.vm->AttachCurrentThread(&threadEnv, 0);
        if (ret < 0) {
            logMessage(aoce::LogLevel::warn,
                       "andorid jni attach thread failed");
            return nullptr;
        }
        bAttach = true;
        logMessage(aoce::LogLevel::info, "andorid jni attach thread success");
    }
    return threadEnv;
}

void AoceManager::detachThread() {
    // 直接DetachCurrentThread可能出问题,先检测是否已经附加到线程
    JNIEnv *threadEnv = nullptr;
    jint ret =
        androidEnv.vm->GetEnv((void **)&threadEnv, androidEnv.sdkVersion);
    if (ret >= 0 && threadEnv != nullptr) {
        if(androidEnv.application){
            threadEnv->DeleteGlobalRef(androidEnv.application);
        }
        androidEnv.vm->DetachCurrentThread();
    }
}

jobject AoceManager::getActivityApplication(jobject activity, JNIEnv *env) {
    jmethodID methodGetApplication =
        env->GetMethodID(env->GetObjectClass(androidEnv.activity),
                         "getApplication", "()Landroid/app/Application;");
    jobject application = env->CallObjectMethod(activity, methodGetApplication);
    return application;
}

std::string AoceManager::getObjClassName(jobject obj, JNIEnv *env) {
    if (env == nullptr) {
        env = androidEnv.env;
    }
    jclass cls = env->GetObjectClass(obj);
    // First get the class object
    jmethodID mid = env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
    jobject clsObj = env->CallObjectMethod(androidEnv.activity, mid);

    // Now get the class object's class descriptor
    cls = env->GetObjectClass(clsObj);
    // Find the getName() method on the class object
    mid = env->GetMethodID(cls, "getName", "()Ljava/lang/String;");

    // Call the getName() to get a jstring object back
    jstring strObj = (jstring)env->CallObjectMethod(clsObj, mid);
    const char *str = env->GetStringUTFChars(strObj, NULL);
    // copy
    std::string result = str;
    env->ReleaseStringUTFChars(strObj, str);
    return str;
}

#endif
}  // namespace aoce