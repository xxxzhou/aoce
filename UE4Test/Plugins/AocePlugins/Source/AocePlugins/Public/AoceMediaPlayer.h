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
	aoce::MediaPlayer* player = nullptr;

	aoce::PipeGraph* vkGraph = nullptr;
	aoce::InputLayer* inputLayer = nullptr;
	aoce::OutputLayer* outputLayer = nullptr;
	aoce::YUV2RGBALayer* yuv2rgbLayer = nullptr;
	aoce::VideoFormat format = {};

	aoce::GpuType gpuType = aoce::GpuType::vulkan;
	bool bFill = true;
	std::vector<std::uint8_t> data;

	AAoceDisplayActor* display = nullptr;
public:
	aoce::MediaPlayer* getPlay();
public:
	UTexture2D * sourceTex = nullptr;
public:
	void initPlay(AAoceDisplayActor* display);
	void updateTexture(UTexture2D** texture, int width, int height, EPixelFormat format = PF_R8G8B8A8);

public:
	virtual void onPrepared() override;
	virtual void onError(aoce::PlayStatus staus, int32_t code,
		std::string msg) override;
	virtual void onVideoFrame(const aoce::VideoFrame& frame) override;
	virtual void onStop() override;
	virtual void onComplate() override;
};
