// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "AocePlugins.h"
#include "Core.h"
#include "Modules/ModuleManager.h"

#define LOCTEXT_NAMESPACE "FAocePluginsModule"

void FAocePluginsModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	LoadDllEx(FString("AocePlugins/ThirdParty/Aoce/win/bin/aoce.dll"), false);	
	// loadAoce();
}

void FAocePluginsModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	// unloadAoce();
}

void FAocePluginsModule::LoadDllEx(FString relativePath, bool bSeacher) {
	FString filePath = FPaths::ProjectPluginsDir() / relativePath;
	if (FPaths::FileExists(filePath))
	{
		FString fullFileName = FPaths::ConvertRelativePathToFull(filePath);
		FString fullPath = FPaths::GetPath(fullFileName);
		void *hdll = nullptr;
		if (bSeacher) {
			//FPlatformProcess::PushDllDirectory(*fullPath);
			//hdll = LoadLibraryEx(*fullFileName, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
			//FPlatformProcess::PopDllDirectory(*fullPath);
		}
		else {
			FPlatformProcess::PushDllDirectory(*fullPath);
			hdll = FPlatformProcess::GetDllHandle(*fullFileName);
			FPlatformProcess::PopDllDirectory(*fullPath);
		}
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