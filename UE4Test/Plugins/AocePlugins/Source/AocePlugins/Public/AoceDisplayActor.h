// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/Texture2D.h"
#include "aoce/AoceCore.h"
#include "AoceDisplayActor.generated.h"

UCLASS()
class AOCEPLUGINS_API AAoceDisplayActor : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	AAoceDisplayActor();
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		class AStaticMeshActor* actor;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		class UMaterialInterface* material;
public:
	UFUNCTION(BlueprintCallable)
		void SetTexture(UTexture2D* texture);
private:
	class UMaterialInstanceDynamic* materialDynamic = nullptr;
private:
	aoce::PipeGraph* vkGraph = nullptr;
	aoce::InputLayer* inputLayer = nullptr;
	aoce::OutputLayer* outputLayer = nullptr;
	aoce::YUV2RGBALayer* yuv2rgbLayer = nullptr;
	aoce::VideoFormat format = {};

	aoce::GpuType gpuType = aoce::GpuType::vulkan;
	FTexture2DRHIRef textureRHI = nullptr;
	UTexture2D* sourceTex = nullptr;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	void outLayerData(uint8_t* data, int32_t width, int32_t height,
		int32_t outIndex);
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
public:
	// 请保证这个方法在游戏线程调用
	void UpdateFrame(const aoce::VideoFrame& frame);
};

void initTexture(UTexture2D** texture, int width, int height, EPixelFormat format = PF_R8G8B8A8);
