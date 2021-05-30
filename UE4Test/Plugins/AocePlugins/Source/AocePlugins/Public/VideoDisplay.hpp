#pragma once

#include "aoce/AoceCore.h"
#include "Engine/Texture2D.h"
#include <functional>

namespace aoce {

	void initTexture(UTexture2D** texture, int width, int height, EPixelFormat format = PF_R8G8B8A8);

	// 用于UE4 Tex与OutLayer交互
	// UE4 android使用opengl,UE4 win32使用dx11
	class VideoDisplay : public IOutputLayerObserver {
	public:
		VideoDisplay(/* args */);
		~VideoDisplay();
	private:
		/* data */
		IOutputLayer* outLayer = nullptr;
		FTexture2DRHIRef textureRHI = nullptr;
		ImageFormat format = {};
		std::function<void(const ImageFormat&)> onTextureChange;
	public:
		UTexture2D* displayTex = nullptr;
	public:
		void setOutLayer(IOutputLayer* layer);
		void setTextureChange(std::function<void(const ImageFormat&)> onTextureChangeAction);
		void changeDisplay();
		void updateFrameGPU();
	public:
		virtual void onImageProcess(uint8_t* data, const ImageFormat& imageFormat,
			int32_t outIndex) override;
		virtual void onFormatChanged(const ImageFormat& imageFormat,
			int32_t outIndex) override;
	};

}  // namespace aoce
