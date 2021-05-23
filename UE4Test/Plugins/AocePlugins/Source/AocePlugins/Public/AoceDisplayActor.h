// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/Texture2D.h"
#include "aoce/AoceCore.h"
#include "VideoDisplay.hpp"
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
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		UTexture2D* sourceTex = nullptr;
private:
	aoce::VideoDisplay* videoDisplay = nullptr;

private:
	class UMaterialInstanceDynamic* materialDynamic = nullptr;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	void onTextureChange(const aoce::ImageFormat imageFormat);
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
public:
	void setDisplay(aoce::VideoDisplay* display);
};

