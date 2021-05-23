// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceLiveActor.h"
#include "LiveManager.hpp"
#include "CameraManager.hpp"

using namespace aoce;

// Sets default values
AAoceLiveActor::AAoceLiveActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AAoceLiveActor::BeginPlay() {
	Super::BeginPlay();
	aoce::LiveManager::Get().initLive(aoce::LiveType::agora, aoce::GpuType::vulkan);
	auto videoList = aoce::CameraManager::Get().getVideoList();
	if (videoList.size() > 0) {
		auto firstVideo = videoList[0];
		int32_t format = firstVideo->findFormatIndex(1280, 720);
		aoce::LiveManager::Get().setCameraId(videoList[0]->getId(), format);
	}
}

void AAoceLiveActor::onLogin(int roleIndex) {
	RoleType rtype = (RoleType)roleIndex;
	int32_t userIndex = 0;
	if (rtype == RoleType::student) {
		userIndex = 5;
	}
	aoce::LiveManager::Get().loginRoom("123", rtype, userIndex);
	dispActor->setDisplay(aoce::LiveManager::Get().getVideoDisplay());
}

void AAoceLiveActor::onLogout() {
	aoce::LiveManager::Get().logoutRoom();
}

void AAoceLiveActor::showDevice(int modeIndex) {
	aoce::CameraProcess *cp = aoce::LiveManager::Get().getCameraProcess();
	if (modeIndex == 1) {
		cp->openCamera(aoce::ProcessType::source);
	}
	else if (modeIndex == 2) {
		cp->openCamera(aoce::ProcessType::matting);
	}
	dispActor->setDisplay(cp->getVideoDisplay());
}

void AAoceLiveActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
	aoce::LiveManager::Get().logoutRoom();
	aoce::LiveManager::Get().shutdownRoom();
}

// Called every frame
void AAoceLiveActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);

}

