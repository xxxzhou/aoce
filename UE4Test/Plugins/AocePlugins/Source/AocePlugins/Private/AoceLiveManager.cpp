// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceLiveManager.h"
#include <functional>
#include "Engine/Engine.h"
#include "Async/Async.h"
// #include "OpenGLDrv.h"
#if PLATFORM_ANDROID
#include <GLES2/gl2.h>
#include "OpenGLUtil.h"
#include "Runtime/Launch/Public/Android/AndroidJNI.h"
#include "Runtime/ApplicationCore/Public/Android/AndroidApplication.h"
#endif
using namespace aoce;
using namespace std::placeholders;

AoceLiveManager::AoceLiveManager()
{
}

AoceLiveManager::~AoceLiveManager()
{
}

void AoceLiveManager::aoceMsg(int32_t level, const char* message) {
	AsyncTask(ENamedThreads::GameThread, [&]() {
		if (GEngine) {
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, message);
		}
	});
}

void AoceLiveManager::initRoom(aoce::LiveType liveType, AAoceDisplayActor* pdisplay)
{
	this->display = pdisplay;
	room = AoceManager::Get().getLiveRoom(liveType);
	AgoraContext contex = {};
	contex.bLoopback = true;
#if __ANDROID__ // PLATFORM_ANDROID
	// contex.context = jcontext;
#endif	
	room->initRoom(&contex, this);
}

aoce::LiveRoom* AoceLiveManager::getRoom() {
	return room;
}

void AoceLiveManager::onEvent(int32_t operater, int32_t code, aoce::LogLevel level, const std::string& msg)
{
}

void AoceLiveManager::onInitRoom()
{
}

void AoceLiveManager::onLoginRoom(bool bReConnect)
{
}

void AoceLiveManager::onUserChange(int32_t userId, bool bAdd)
{
	std::string str;
	string_format(str, (bAdd ? "add" : "remove"), "user: ", userId);
	logMessage(AOCE_LOG_INFO, str.c_str());
	if (bAdd) {
		PullSetting setting = {};
		room->pullStream(userId, 0, setting);
	}
	else {
		room->stopPullStream(userId, 0);
	}
}

void AoceLiveManager::onStreamUpdate(int32_t index, bool bAdd, int32_t code)
{
	//logMessage(AOCE_LOG_INFO, "xxxxxxx4");
}

void AoceLiveManager::onStreamUpdate(int32_t userId, int32_t index, bool bAdd, int32_t code)
{
	//logMessage(AOCE_LOG_INFO, "xxxxxxx5");
}

void AoceLiveManager::onVideoFrame(int32_t userId, int32_t index, const aoce::VideoFrame& videoFrame)
{
	// logMessage(AOCE_LOG_INFO, "xxxxxxx6");
	AsyncTask(ENamedThreads::GameThread, [=]() {
		display->UpdateFrame(videoFrame);
	});
}

void AoceLiveManager::onAudioFrame(int32_t userId, int32_t index, const aoce::AudioFrame& audioFrame)
{
}

void AoceLiveManager::onPushQuality(int32_t index, int32_t quality, float fps, float kbs)
{
}

void AoceLiveManager::onPullQuality(int32_t userId, int32_t index, int32_t quality, float fps, float kbs)
{
}

void AoceLiveManager::onLogoutRoom()
{
}
