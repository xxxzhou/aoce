// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
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
		AAoceDisplayActor* dispActor;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
public:
	UFUNCTION(BlueprintCallable)
		void onLogin(int roleIndex);
	UFUNCTION(BlueprintCallable)
		void onLogout();
	UFUNCTION(BlueprintCallable)
		void showDevice(int modeIndex);
	UFUNCTION(BlueprintCallable)
		void onChangeDisplay();
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
