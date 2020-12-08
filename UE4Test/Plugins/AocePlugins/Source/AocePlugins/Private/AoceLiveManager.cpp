// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceLiveManager.h"
#include <functional>
#include "Engine/Engine.h"
#if PLATFORM_ANDROID
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
	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, message);
	}
}

void AoceLiveManager::initRoom(aoce::LiveType liveType)
{
	// setLogHandle(std::bind(&AoceLiveManager::aoceMsg, this, _1, _2));	
	logMessage(AOCE_LOG_INFO, "xxxxxxx1");
	// 生成一张执行图
	auto graphFactory = AoceManager::Get().getPipeGraphFactory(gpuType);	
	if (graphFactory == nullptr) {
		logMessage(AOCE_LOG_INFO, "no graph factory");
		return;
	}
	logMessage(AOCE_LOG_INFO, "have graph factory");
	vkGraph = graphFactory->createGraph();
	auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
	inputLayer = layerFactory->crateInput();
	outputLayer = layerFactory->createOutput();
	// 输出GPU数据
	outputLayer->updateParamet({ false, true });
	yuv2rgbLayer = layerFactory->createYUV2RGBA();
	// 生成图
	vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
	room = AoceManager::Get().getLiveRoom(liveType);
	AgoraContext contex = {};
	contex.bLoopback = false;
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
	logMessage(AOCE_LOG_INFO, "xxxxxxx4");
}

void AoceLiveManager::onStreamUpdate(int32_t userId, int32_t index, bool bAdd, int32_t code)
{
	logMessage(AOCE_LOG_INFO, "xxxxxxx5");
}

void AoceLiveManager::onVideoFrame(int32_t userId, int32_t index, const aoce::VideoFrame& videoFrame)
{
	logMessage(AOCE_LOG_INFO, "xxxxxxx6");
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
