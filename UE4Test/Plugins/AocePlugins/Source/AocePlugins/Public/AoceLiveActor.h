// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AoceLiveManager.h"
#include "AoceMediaPlayer.h"
#include <memory>
#include "AoceDisplayActor.h"
#include "AoceLiveActor.generated.h"

UCLASS()
class AOCEPLUGINS_API AAoceLiveActor : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	AAoceLiveActor();
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		AAoceDisplayActor *dispActor;
private:
	std::unique_ptr<AoceLiveManager> liveManager = nullptr;
	std::unique_ptr<AoceMediaPlayer> mediaPlayer = nullptr;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
