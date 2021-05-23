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

// Sets default values
AAoceDisplayActor::AAoceDisplayActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AAoceDisplayActor::BeginPlay() {
	Super::BeginPlay();

	materialDynamic = UMaterialInstanceDynamic::Create(material, this);
	actor->GetStaticMeshComponent()->SetMaterial(0, materialDynamic);
}

void AAoceDisplayActor::onTextureChange(const aoce::ImageFormat imageFormat) {
	if (videoDisplay && materialDynamic) {
		materialDynamic->SetTextureParameterValue("mainTex", videoDisplay->displayTex);
		sourceTex = videoDisplay->displayTex;
	}
}

// Called every frame
void AAoceDisplayActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	if (videoDisplay) {
		videoDisplay->updateFrameGPU();
	}
}

void AAoceDisplayActor::setDisplay(aoce::VideoDisplay* display) {
	if (videoDisplay != nullptr) {
		videoDisplay->setTextureChange(nullptr);
	}
	videoDisplay = display;
	videoDisplay->setTextureChange(std::bind(&AAoceDisplayActor::onTextureChange, this, _1));
}



