// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceLiveActor.h"
#if PLATFORM_ANDROID
#include "Runtime/Launch/Public/Android/AndroidJNI.h"
#include "Runtime/ApplicationCore/Public/Android/AndroidApplication.h"
#endif

using namespace aoce;

// Sets default values
AAoceLiveActor::AAoceLiveActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AAoceLiveActor::BeginPlay()
{
	Super::BeginPlay();
	loadAoce();
#if __ANDROID__ // PLATFORM_ANDROID
	JNIEnv* jni_env = FAndroidApplication::GetJavaEnv(true);
	jobject at = FAndroidApplication::GetGameActivityThis();
	jmethodID getApplication = jni_env->GetMethodID(FJavaWrapper::GameActivityClassID, "getApplication", "()Landroid/app/Application;");
	jobject jcontext = jni_env->CallObjectMethod(at, getApplication);
	AndroidEnv andEnv = {};
	andEnv.env = jni_env;
	andEnv.activity = at;
	//// 从https://blog.csdn.net/hust_liuX/article/details/1460486 得到的修改方案
	//JavaVM* g_jvm = nullptr;
	//jni_env->GetJavaVM(&g_jvm);
	//g_jvm->AttachCurrentThread(&jni_env, 0);
	AoceManager::Get().initAndroid(andEnv);
#endif	
	//liveManager = std::make_unique<AoceLiveManager>();
	//liveManager->initRoom(aoce::LiveType::agora);
	//liveManager->getRoom()->loginRoom("123", 16, 0);

	std::string uri = "rtmp://58.200.131.2:1935/livetv/hunantv";
	mediaPlayer = std::make_unique< AoceMediaPlayer>();
	mediaPlayer->initPlay(dispActor);
	mediaPlayer->getPlay()->setDataSource(uri.c_str());
	mediaPlayer->getPlay()->prepare(false);
}

void AAoceLiveActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
	//liveManager->getRoom()->shutdownRoom();
	mediaPlayer->getPlay()->stop();
	unloadAoce();
}

// Called every frame
void AAoceLiveActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

