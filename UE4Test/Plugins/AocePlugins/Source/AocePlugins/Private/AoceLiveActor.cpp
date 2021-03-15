// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceLiveActor.h"
#if PLATFORM_ANDROID
#include "Runtime/Launch/Public/Android/AndroidJNI.h"
#include "Runtime/ApplicationCore/Public/Android/AndroidApplication.h"
#endif

using namespace aoce;

// Sets default values
AAoceLiveActor::AAoceLiveActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AAoceLiveActor::BeginPlay() {
	Super::BeginPlay();	

	liveManager = std::make_unique<AoceLiveManager>();
	liveManager->initRoom(aoce::LiveType::agora, dispActor);
	liveManager->getRoom()->loginRoom("12345", 16, 0);

//	std::string uri = "rtmp://58.200.131.2:1935/livetv/hunantv";
//	mediaPlayer = std::make_unique< AoceMediaPlayer>();
//	mediaPlayer->initPlay(dispActor);
//	mediaPlayer->getPlay()->setDataSource(uri.c_str());
//#if WIN32
//	mediaPlayer->getPlay()->prepare(true);
//#elif __ANDROID__
//	mediaPlayer->getPlay()->prepare(false);
//#endif
}

void AAoceLiveActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
	if (liveManager) {
		liveManager->getRoom()->logoutRoom();
		liveManager->getRoom()->shutdownRoom();
	}
	if (mediaPlayer) {
		mediaPlayer->getPlay()->stop();
	}	
}

// Called every frame
void AAoceLiveActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	if (mediaPlayer) {
		// mediaPlayer->updateTexture();
	}
}

