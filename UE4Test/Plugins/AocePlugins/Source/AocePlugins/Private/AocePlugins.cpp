// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "AocePlugins.h"
#include "Core.h"
#include "aoce/AoceCore.h"
#include "Modules/ModuleManager.h"
#if __ANDROID__
#include "Runtime/Launch/Public/Android/AndroidJNI.h"
#include "Runtime/ApplicationCore/Public/Android/AndroidApplication.h"
#endif

#define LOCTEXT_NAMESPACE "FAocePluginsModule"

void FAocePluginsModule::StartupModule()
{
#if WIN32
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	// LoadDllEx(FString("AocePlugins/ThirdParty/Aoce/win/bin/aoce.dll"), false);
	FString relativePath = FString("AocePlugins/ThirdParty/Aoce/win/bin/aoce.dll");
	FString filePath = FPaths::ProjectPluginsDir() / relativePath;
	if (FPaths::FileExists(filePath))
	{
		FString fullFileName = FPaths::ConvertRelativePathToFull(filePath);
		FString fullPath = FPaths::GetPath(fullFileName);
		FPlatformProcess::PushDllDirectory(*fullPath);
		void* hdll = FPlatformProcess::GetDllHandle(*fullFileName);
		FPlatformProcess::PopDllDirectory(*fullPath);
		if (hdll == nullptr) {
		}
	}
#endif	
	aoce::loadAoce();
#if __ANDROID__ // PLATFORM_ANDROID __ANDROID__
	JNIEnv* jni_env = FAndroidApplication::GetJavaEnv(true);
	jobject at = FAndroidApplication::GetGameActivityThis();
	jmethodID getApplication = jni_env->GetMethodID(FJavaWrapper::GameActivityClassID, "getApplication", "()Landroid/app/Application;");
	jobject jcontext = jni_env->CallObjectMethod(at, getApplication);
	AndroidEnv andEnv = {};
	andEnv.env = jni_env;
	andEnv.activity = at;
	initAndroid(andEnv);
#endif	
}

void FAocePluginsModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	aoce::unloadAoce();
}

void FAocePluginsModule::LoadDllEx(FString relativePath, bool bSeacher) {
	FString filePath = FPaths::ProjectPluginsDir() / relativePath;
	if (FPaths::FileExists(filePath))
	{
		FString fullFileName = FPaths::ConvertRelativePathToFull(filePath);
		FString fullPath = FPaths::GetPath(fullFileName);
		FPlatformProcess::PushDllDirectory(*fullPath);
		void* hdll = FPlatformProcess::GetDllHandle(*fullFileName);
		FPlatformProcess::PopDllDirectory(*fullPath);
		if (hdll == nullptr) {
			//int error_id = GetLastError();
			//if (error_id != 0)
			//{
			//	FString fileName = FPaths::GetBaseFilename(fullPath);
			//	FString message = "load " + fileName + " error:" + FString::FromInt(error_id);
			//	UE_LOG(LogTemp, Error, TEXT("%s"), *message);
			//}
		}
	}
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FAocePluginsModule, AocePlugins)