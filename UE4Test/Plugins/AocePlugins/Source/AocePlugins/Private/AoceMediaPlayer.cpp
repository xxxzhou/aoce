// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceMediaPlayer.h"
#include "Async/Async.h"
using namespace aoce;

AoceMediaPlayer::AoceMediaPlayer()
{
	player = AoceManager::Get().getMediaPlayer(MediaPlayType::ffmpeg);
	player->setObserver(this);
}

AoceMediaPlayer::~AoceMediaPlayer()
{
}

aoce::MediaPlayer* AoceMediaPlayer::getPlay()
{
	return player;
}

void AoceMediaPlayer::initPlay(AAoceDisplayActor* pdisplay) {
	this->display = pdisplay;
	// 生成一张执行图
	vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
	auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
	inputLayer = layerFactory->crateInput();
	outputLayer = layerFactory->createOutput();
	// 输出GPU数据
	outputLayer->updateParamet({ true, false });
	yuv2rgbLayer = layerFactory->createYUV2RGBA();
	// 生成图
	vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
}

void AoceMediaPlayer::updateTexture(UTexture2D** ptexture, int width, int height, EPixelFormat format)
{
	UTexture2D* texture = *ptexture;
	bool bValid = texture && texture->IsValidLowLevel();
	bool bChange = false;
	if (bValid) {
		int twidth = texture->GetSizeX();
		int theight = texture->GetSizeY();
		bChange = (twidth != width) || (theight != height);
		if (bChange) {
			texture->RemoveFromRoot();
			texture->ConditionalBeginDestroy();
			texture = nullptr;
		}
	}
	if (!bValid || bChange) {
		*ptexture = UTexture2D::CreateTransient(width, height, format);
		// (*ptexture)->MipGenSettings = TMGS_NoMipmaps;
		(*ptexture)->UpdateResource();
		(*ptexture)->AddToRoot();
	}
}

void AoceMediaPlayer::onPrepared()
{
	logMessage(AOCE_LOG_INFO, "media prepared");
	player->start();
	stop = false;
}

void AoceMediaPlayer::onError(aoce::PlayStatus staus, int32_t code, std::string msg)
{
}

void AoceMediaPlayer::onVideoFrame(const aoce::VideoFrame& frame)
{
	if (stop) {
		return;
	}
	logMessage(AOCE_LOG_INFO, "media frame");
	AsyncTask(ENamedThreads::GameThread, [=]() {
		if (stop) {
			return;
		}
		if (format.width != frame.width || format.height != frame.height) {
			format.width = frame.width;
			format.height = frame.height;
			format.videoType = frame.videoType;
			inputLayer->setImage(format);
			yuv2rgbLayer->updateParamet({ format.videoType });
			int32_t size = getVideoFrame(frame, nullptr);
			bFill = size == 0;
			if (!bFill) {
				data.resize(size);
			}
			updateTexture(&sourceTex, frame.width, frame.height, PF_R8G8B8A8);
			display->SetTexture(sourceTex);
		}
		AoceMediaPlayer* play = this;
		ENQUEUE_RENDER_COMMAND(CopyTextureCommand)([play](FRHICommandListImmediate& RHICmdList) {
			// void* device = RHICmdList.GetNativeDevice();
			RHICmdList.GetNativeDevice();
			void* sourceResource = play->sourceTex->Resource->TextureRHI->GetNativeResource();
			VkOutGpuTex outGpuTex = {};
#if WIN32
			outGpuTex.image = sourceResource;
#elif __ANDROID__
			int32 textid = *reinterpret_cast<int32*>(sourceResource);			
			outGpuTex.image = textid;
			outGpuTex.width = play->format.width;
			outGpuTex.height = play->format.height;
			// GL_TEXTURE_2D_MULTISAMPLE(0x9100)/GL_TEXTURE_2D(0x0DE1)/GL_TEXTURE_EXTERNAL_OES/GL_TEXTURE_RECTANGLE(0x84F5)
			play->outputLayer->outGLGpuTex(outGpuTex, 0x9100);
#endif					
		});
		if (bFill) {
			inputLayer->inputCpuData(frame.data[0], 0);
		}
		else {
			getVideoFrame(frame, data.data());
			inputLayer->inputCpuData(data.data(), 0);
		}
		vkGraph->run();
	});

}

void AoceMediaPlayer::onStop()
{
}

void AoceMediaPlayer::onComplate()
{
	stop = true;
}
