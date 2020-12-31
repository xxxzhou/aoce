// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "aoce/AoceCore.h"
#include "Engine/Texture2D.h"
#include "AoceDisplayActor.h"

/**
 *
 */
class AOCEPLUGINS_API AoceMediaPlayer : public aoce::IMediaPlayerObserver
{
public:
	AoceMediaPlayer();
	virtual ~AoceMediaPlayer() override;

private:
	int index = 0;
	aoce::MediaPlayer* player = nullptr;
	AAoceDisplayActor* display = nullptr;
	bool bStart = false;
public:
	aoce::MediaPlayer* getPlay();
public:
	UTexture2D* sourceTex = nullptr;
public:
	void initPlay(AAoceDisplayActor* display);	
public:
	virtual void onPrepared() override;
	virtual void onError(aoce::PlayStatus staus, int32_t code,
		std::string msg) override;
	virtual void onVideoFrame(const aoce::VideoFrame& frame) override;
	virtual void onStop() override;
	virtual void onComplate() override;
};
