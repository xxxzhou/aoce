%module(directors = "1") AoceWrapper
%{
#include "aoce/Aoce.h"
#include "aoce/AoceCore.h"
#include "aoce/AoceDefine.h"
%}

#define ACOE_EXPORT

%include "stdint.i"

%include "aoce/Aoce.h"
%include "aoce/AoceLayer.h"
%include "aoce/AoceLive.h"
%include "aoce/AoceMath.h"
%include "aoce/AoceMedia.h"
%include "aoce/AoceVideoDevice.h"
%include "aoce/AoceCore.h"

namespace aoce {
    
%feature("director") IMediaPlayerObserver;
%feature("director") IBaseLayer;
%feature("director") PipeGraphFactory;
%feature("director") LayerFactory;
%feature("director") IVideoManager;
%feature("director") MediaFactory;
%feature("director") ILiveRoom;
%feature("director") IVideoDevice;
    
// void loadAoce();
// void unloadAoce();
// bool checkLoadModel(const char* modelName);
// aoce::PipeGraphFactory* getPipeGraphFactory(const aoce::GpuType& gpuType);
// aoce::LayerFactory* getLayerFactory(const aoce::GpuType& gpuType);
// aoce::IVideoManager* getVideoManager(const aoce::CameraType& cameraType);
// aoce::MediaFactory* getMediaFactory(
//     const aoce::MediaType& mediaType);
// aoce::ILiveRoom* getLiveRoom(const aoce::LiveType& liveType);

%nodefaultctor;
%nodefaultdtor;


%clearnodefaultctor;
%clearnodefaultdtor;


};  // namespace aoce




