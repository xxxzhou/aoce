#include "AoceCore.h"

#include "AoceManager.hpp"
#include "module/ModuleManager.hpp"

namespace aoce {
 
bool checkLoadModel(const char* modelName) {
    return ModuleManager::Get().checkLoadModel(modelName);
}

PipeGraphFactory* getPipeGraphFactory(const GpuType& gpuType) {
    auto* factory = AoceManager::Get().getPipeGraphFactory(gpuType);
    return factory;
}

LayerFactory* getLayerFactory(const GpuType& gpuType) {
    auto layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    return layerFactory;
}

IVideoManager* getVideoManager(const CameraType& cameraType) {
    IVideoManager* manager = AoceManager::Get().getVideoManager(cameraType);
    return manager;
}

MediaFactory* getMediaFactory(const MediaType& mediaType) {
    auto factory = AoceManager::Get().getMediaFactory(mediaType);
    return factory;
}

ILiveRoom* getLiveRoom(const LiveType& liveType) {
    LiveRoom* room = AoceManager::Get().getLiveRoom(liveType);
    return room;
}

}  // namespace aoce