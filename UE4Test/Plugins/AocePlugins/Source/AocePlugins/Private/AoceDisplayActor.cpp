// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceDisplayActor.h"
#include "Async/Async.h"
#include "Materials/MaterialInstance.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"

#if __ANDROID__
#include <GLES2/gl2.h>
#include "OpenGLUtil.h"
#endif

using namespace std::placeholders;
using namespace aoce;

static void UpdateAoceGLTexture(aoce::OutputLayer* outLayer, FTexture2DRHIRef textRef, int width, int height) {
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

struct FRHICommandUpdateAoceGLTexture final : public FRHICommand< FRHICommandUpdateAoceGLTexture>
{
	FTexture2DRHIRef textRef = nullptr;
	aoce::OutputLayer* outLayer = nullptr;
	int width = 0;
	int height = 0;

	FORCEINLINE_DEBUGGABLE FRHICommandUpdateAoceGLTexture(aoce::OutputLayer* inOutLayer, FTexture2DRHIRef inTextRef, int inWidth, int inHeight) {
		outLayer = inOutLayer;
		textRef = inTextRef;
		width = inWidth;
		height = inHeight;
	}

	void Execute(FRHICommandListBase& CmdList) {
		UpdateAoceGLTexture(outLayer, textRef, width, height);
	}
};


// Sets default values
AAoceDisplayActor::AAoceDisplayActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void AAoceDisplayActor::SetTexture(UTexture2D* texture) {
	if (materialDynamic) {
		materialDynamic->SetTextureParameterValue("mainTex", texture);
	}
}

// Called when the game starts or when spawned
void AAoceDisplayActor::BeginPlay() {
	Super::BeginPlay();

	materialDynamic = UMaterialInstanceDynamic::Create(material, this);
	actor->GetStaticMeshComponent()->SetMaterial(0, materialDynamic);

	// 生成一张执行图
	vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
	auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
	inputLayer = layerFactory->crateInput();
	outputLayer = layerFactory->createOutput();
	// 输出GPU数据
	outputLayer->updateParamet({ true, false });
	// outputLayer->setImageProcessHandle(std::bind(&AoceMediaPlayer::outLayerData, this, _1, _2, _3, _4));
	yuv2rgbLayer = layerFactory->createYUV2RGBA();
	// 生成图
	vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
}

void AAoceDisplayActor::outLayerData(uint8_t* data, int32_t width, int32_t height, int32_t outIndex)
{
#if __ANDROID__
	AAoceDisplayActor* play = this;
	ENQUEUE_RENDER_COMMAND(CopyTextureCommand)([play, data](FRHICommandListImmediate& RHICmdList) {
		FUpdateTextureRegion2D region = {};
		region.Width = play->format.width;
		region.Height = play->format.height;
		region.DestX = 0;
		region.DestY = 0;
		region.SrcX = 0;
		region.SrcY = 0;
		RHIUpdateTexture2D(play->textureRHI, 0, region, play->format.width * 4, data);
	});
#endif
}

// Called every frame
void AAoceDisplayActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
}

void AAoceDisplayActor::UpdateFrame(const aoce::VideoFrame& frame) {
	AAoceDisplayActor* play = this;
	if (format.width != frame.width || format.height != frame.height) {
		format.width = frame.width;
		format.height = frame.height;
		format.videoType = frame.videoType;
		inputLayer->setImage(format);
		yuv2rgbLayer->updateParamet({ format.videoType });
		initTexture(&sourceTex, frame.width, frame.height, PF_R8G8B8A8);
		SetTexture(sourceTex);
		ENQUEUE_RENDER_COMMAND(CreateTextureCommand)([play](FRHICommandListImmediate& RHICmdList) {
			if (play->sourceTex->IsValidLowLevel()) {
				play->textureRHI = ((FTexture2DResource*)(play->sourceTex->Resource))->GetTexture2DRHI();
			}
			else {
				play->textureRHI = nullptr;
			}
		});
	}
	inputLayer->inputCpuData(frame, 0);
	vkGraph->run();
	ENQUEUE_RENDER_COMMAND(CopyTextureCommand)([play](FRHICommandListImmediate& RHICmdList) {
		FTexture2DRHIRef texRHI = play->textureRHI;// ((FTexture2DResource*)(play->sourceTex->Resource))->GetTexture2DRHI();
		if (!texRHI) {
			return;
		}
		if (IsRunningRHIInSeparateThread()) {
			new (RHICmdList.AllocCommand<FRHICommandUpdateAoceGLTexture>()) FRHICommandUpdateAoceGLTexture(play->outputLayer, texRHI, play->format.width, play->format.height);
			RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		}
		else {
			UpdateAoceGLTexture(play->outputLayer, texRHI, play->format.width, play->format.height);
		}
	});
}

void initTexture(UTexture2D** ptexture, int width, int height, EPixelFormat format)
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
		// UTexture2DExternal
		*ptexture = UTexture2D::CreateTransient(width, height, format);
		// (*ptexture)->MipGenSettings = TMGS_NoMipmaps;
		// (*ptexture)->CompressionSettings = TextureCompressionSettings::TC_VectorDisplacementmap;;
		// (*ptexture)->SRGB = false;
		(*ptexture)->UpdateResource();
		(*ptexture)->AddToRoot();
	}
}

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

