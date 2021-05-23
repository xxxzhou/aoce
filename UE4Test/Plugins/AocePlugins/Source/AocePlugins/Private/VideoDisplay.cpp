#include "VideoDisplay.hpp"
#if PLATFORM_ANDROID
#include "Runtime/Launch/Public/Android/AndroidJNI.h"
#include "Runtime/ApplicationCore/Public/Android/AndroidApplication.h"
#endif

namespace aoce {

	static void UpdateAoceGLTexture(aoce::IOutputLayer* outLayer, FTexture2DRHIRef textRef, int width, int height) {
		if (outLayer == nullptr || !textRef.IsValid()) {
			return;
		}
		void* sourceResource = textRef->GetNativeResource();
		VkOutGpuTex outGpuTex = {};
#if WIN32
		outGpuTex.image = sourceResource;
#elif __ANDROID__
		//outGpuTex.commandbuffer = dataPtr;
		int32 textid = *reinterpret_cast<int32*>(sourceResource);
		outGpuTex.image = textid;
#endif
		outGpuTex.width = width;
		outGpuTex.height = height;
#if __ANDROID__
		outLayer->outGLGpuTex(outGpuTex);
#endif
	}

	struct FRHICommandUpdateAoceGLTexture final : public FRHICommand< FRHICommandUpdateAoceGLTexture> {
		FTexture2DRHIRef textRef = nullptr;
		aoce::IOutputLayer* outLayer = nullptr;
		int width = 0;
		int height = 0;

		FORCEINLINE_DEBUGGABLE FRHICommandUpdateAoceGLTexture(aoce::IOutputLayer* inOutLayer, FTexture2DRHIRef inTextRef, int inWidth, int inHeight) {
			outLayer = inOutLayer;
			textRef = inTextRef;
			width = inWidth;
			height = inHeight;
		}

		void Execute(FRHICommandListBase& CmdList) {
			UpdateAoceGLTexture(outLayer, textRef, width, height);
		}
	};

	void initTexture(UTexture2D** ptexture, int width, int height, EPixelFormat format) {
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
			// UTexture2DExternal
			*ptexture = UTexture2D::CreateTransient(width, height, format);
			// (*ptexture)->MipGenSettings = TMGS_NoMipmaps;
			// (*ptexture)->CompressionSettings = TextureCompressionSettings::TC_VectorDisplacementmap;;
			(*ptexture)->SRGB = true;
			(*ptexture)->UpdateResource();
			(*ptexture)->AddToRoot();
		}
	}

	VideoDisplay::VideoDisplay(/* args */) {}

	VideoDisplay::~VideoDisplay() {
		format = {};
	}

	void VideoDisplay::setOutLayer(IOutputLayer* layer) {
		outLayer = layer;
		outLayer->setObserver(this);
	}

	void VideoDisplay::setTextureChange(std::function<void(const ImageFormat&)> onTextureChangeAction) {
		this->onTextureChange = onTextureChangeAction;
	}

	void VideoDisplay::updateFrameGPU() {
		if (this->outLayer == nullptr || !this->outLayer->getParamet().bGpu) {
			return;
		}
		VideoDisplay* play = this;
		ENQUEUE_RENDER_COMMAND(CopyTextureCommand)([play](FRHICommandListImmediate& RHICmdList) {
			FTexture2DRHIRef texRHI = play->textureRHI;// ((FTexture2DResource*)(play->sourceTex->Resource))->GetTexture2DRHI();
			if (!texRHI) {
				return;
			}
#if WIN32
			void* device = RHICmdList.GetNativeDevice();
			play->outLayer->outDx11GpuTex(device, texRHI->GetNativeResource());
#elif __ANDROID__
			if (IsRunningRHIInSeparateThread()) {
				new (RHICmdList.AllocCommand<FRHICommandUpdateAoceGLTexture>()) FRHICommandUpdateAoceGLTexture(play->outLayer, texRHI, play->format.width, play->format.height);
				RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
			}
			else {
				UpdateAoceGLTexture(play->outLayer, texRHI, play->format.width, play->format.height);
			}
#endif
		});
	}

	void VideoDisplay::onImageProcess(uint8_t* data, const ImageFormat& imageFormat,
		int32_t outIndex) {
		if (this->outLayer == nullptr || !this->outLayer->getParamet().bCpu) {
			return;
		}
		VideoDisplay* play = this;
		ENQUEUE_RENDER_COMMAND(CopyTextureCommand)([play, data](FRHICommandListImmediate& RHICmdList) {
			if (!play->textureRHI) {
				return;
			}
			FUpdateTextureRegion2D region = {};
			region.Width = play->format.width;
			region.Height = play->format.height;
			region.DestX = 0;
			region.DestY = 0;
			region.SrcX = 0;
			region.SrcY = 0;
			RHIUpdateTexture2D(play->textureRHI, 0, region, play->format.width * 4, data);
		});
	}

	void VideoDisplay::onFormatChanged(const ImageFormat& imageFormat,
		int32_t outIndex) {
		format = imageFormat;
		AsyncTask(ENamedThreads::GameThread, [=]() {
			initTexture(&displayTex, format.width, format.height, PF_R8G8B8A8);
			if (onTextureChange) {
				onTextureChange(format);
			}
			VideoDisplay* play = this;
			ENQUEUE_RENDER_COMMAND(CreateTextureCommand)([play](FRHICommandListImmediate& RHICmdList) {
				if (play->displayTex->IsValidLowLevel()) {
					play->textureRHI = ((FTexture2DResource*)(play->displayTex->Resource))->GetTexture2DRHI();
#if __ANDROID__
					UpdateAoceGLTexture(play->outLayer, play->textureRHI, play->format.width, play->format.height);
#endif
				}
				else {
					play->textureRHI = nullptr;
				}
			});
		});
	}
}  // namespace aoce

//for (uint32 row = 0; row < play->textureRHI->GetSizeY(); row++) {
//	uint8_t* rowPtr = dataPtr;
//	for (uint32 col = 0; col < play->textureRHI->GetSizeX(); col++) {
//		*(rowPtr + 4 * col) = play->index;
//		*(rowPtr + 4 * col + 1) = play->index;
//		*(rowPtr + 4 * col + 2) = play->index;
//		*(rowPtr + 4 * col + 3) = play->index;
//	}
//	dataPtr += rowStride;
//}
// FRHICopyTextureInfo copyInfo = {};
// RHICmdList.CopyTexture(play->textureRHI, texRHI, copyInfo);

//				ALLOC_COMMAND_CL(RHICmdList, FRHICommandGLCommand)([=]() {
//					void* sourceResource = play->textureRHI->GetNativeResource();
//					VkOutGpuTex outGpuTex = {};
//#if WIN32
//					outGpuTex.image = sourceResource;
//#elif __ANDROID__
//					//outGpuTex.commandbuffer = dataPtr;
//					int32 textid = *reinterpret_cast<int32*>(sourceResource);
//
//					outGpuTex.image = textid;
//					outGpuTex.width = play->format.width;
//					outGpuTex.height = play->format.height;
//					// GL_TEXTURE_2D_MULTISAMPLE(0x9100)/GL_TEXTURE_2D(0x0DE1)/GL_TEXTURE_EXTERNAL_OES(0x8D65)/GL_TEXTURE_RECTANGLE(0x84F5)
//					play->outputLayer->outGLGpuTex(outGpuTex);
//#endif
//				});

//				uint32 rowStride = 0;
//				uint8_t* dataPtr = (uint8_t*)RHICmdList.LockTexture2D(texRHI, 0, EResourceLockMode::RLM_WriteOnly, rowStride, false);
//				void* sourceResource = texRHI->GetNativeResource();
//				VkOutGpuTex outGpuTex = {};
//#if WIN32
//				outGpuTex.image = sourceResource;
//#elif __ANDROID__
//				outGpuTex.commandbuffer = dataPtr;
//				int32 textid = *reinterpret_cast<int32*>(sourceResource);
//				outGpuTex.image = textid;
//				outGpuTex.width = play->format.width;
//				outGpuTex.height = play->format.height;
//				// GL_TEXTURE_2D_MULTISAMPLE(0x9100)/GL_TEXTURE_2D(0x0DE1)/GL_TEXTURE_EXTERNAL_OES(0x8D65)/GL_TEXTURE_RECTANGLE(0x84F5)
//				play->outputLayer->outGLGpuTex(outGpuTex);
//#endif		
//				RHICmdList.UnlockTexture2D(texRHI, 0, false);
