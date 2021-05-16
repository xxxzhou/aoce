// Fill out your copyright notice in the Description page of Project Settings.

#include "AoceMediaPlayer.h"

#include "Async/Async.h"
// #include "OpenGLDrv.h"
#if __ANDROID__
#include <GLES2/gl2.h>

#include "OpenGLUtil.h"
#endif

using namespace aoce;

AoceMediaPlayer::AoceMediaPlayer() {
	player = AoceManager::Get().getMediaFactory(MediaType::ffmpeg)->createPlay();
	player->setObserver(this);
}

AoceMediaPlayer::~AoceMediaPlayer() {}

aoce::MediaPlayer* AoceMediaPlayer::getPlay() { return player; }

void AoceMediaPlayer::initPlay(AAoceDisplayActor* pdisplay) {
	this->display = pdisplay;
}

void AoceMediaPlayer::onPrepared() {
	logMessage(AOCE_LOG_INFO, "media prepared");
	player->start();
	bStart = true;
}

void AoceMediaPlayer::onError(aoce::PlayStatus staus, int32_t code,
	std::string msg) {}

void AoceMediaPlayer::onVideoFrame(const aoce::VideoFrame& frame) {
	if (!bStart) {
		return;
	}
	display->UpdateFrame(frame);
}

void AoceMediaPlayer::onStop() {}

void AoceMediaPlayer::onComplate() { bStart = false; }
