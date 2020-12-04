// Fill out your copyright notice in the Description page of Project Settings.


#include "AoceDisplayActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"

// Sets default values
AAoceDisplayActor::AAoceDisplayActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void AAoceDisplayActor::SetTexture(UTexture2D * texture) {
	if (materialDynamic)
		materialDynamic->SetTextureParameterValue("mainTex", texture);
}

// Called when the game starts or when spawned
void AAoceDisplayActor::BeginPlay()
{
	Super::BeginPlay();

	materialDynamic = UMaterialInstanceDynamic::Create(material, this);
	actor->GetStaticMeshComponent()->SetMaterial(0, materialDynamic);
}

// Called every frame
void AAoceDisplayActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

