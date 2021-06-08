#include "LayerMetadata.hpp"

namespace aoce {

#define AOCE_LAYER_PARAMET_NAME "paramet"

BGroupMetadata::BGroupMetadata(const char* paramet_class) {
    parametClass = paramet_class;
}

BGroupMetadata::~BGroupMetadata(){
    
}

int32_t BGroupMetadata::getCount() { return metadatas.size(); }

ILMetadata* BGroupMetadata::getLMetadata(int32_t index) {
    assert(index > 0 && index < metadatas.size());

    return metadatas[index].get();
}

void BGroupMetadata::addMetadata(const char* parametName, const char* text,
                                 bool defaultValue) {
    auto metadata = std::make_shared<LBoolMetadata>();
    metadata->parametName = parametName;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    metadatas.push_back(metadata);
}

void BGroupMetadata::addMetadata(const char* parametName, const char* text,
                                 const char* defaultValue) {
    auto metadata = std::make_shared<LBoolMetadata>();
    metadata->parametName = parametName;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    metadatas.push_back(metadata);
}

void BGroupMetadata::addMetadata(const char* parametName, const char* text,
                                 int32_t defaultValue, int32_t minValue,
                                 int32_t maxVluae) {
    auto metadata = std::make_shared<LIntMetadata>();
    metadata->parametName = parametName;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    metadatas.push_back(metadata);
    metadata->minValue = minValue;
    metadata->maxVluae = maxVluae;
    metadatas.push_back(metadata);
}

void BGroupMetadata::addMetadata(const char* parametName, const char* text,
                                 float defaultValue, float minValue,
                                 float maxVluae) {
    auto metadata = std::make_shared<LFloatMetadata>();
    metadata->parametName = parametName;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    metadatas.push_back(metadata);
    metadata->minValue = minValue;
    metadata->maxVluae = maxVluae;
    metadatas.push_back(metadata);
}

BGroupMetadata* BGroupMetadata::addGroupMetadata(const char* parametName,
                                                 const char* parametClass,
                                                 const char* text) {
    auto metadata = std::make_shared<BGroupMetadata>(parametClass);
    metadata->parametName = parametName;
    metadata->text = text;
    metadatas.push_back(metadata);
    return (BGroupMetadata*)metadata.get();
}

LayerMetadata::LayerMetadata(/* args */) {}

LayerMetadata::~LayerMetadata() {}

LayerMetadataManager* LayerMetadataManager::instance = nullptr;

LayerMetadataManager& LayerMetadataManager::Get() {
    if (instance == nullptr) {
        instance = new LayerMetadataManager();
    }
    return *instance;
}

LayerMetadataManager::LayerMetadataManager(/* args */) {}

LayerMetadataManager::~LayerMetadataManager() {}

void LayerMetadataManager::clean() { instance->layerMetadatas.clear(); }

void LayerMetadataManager::addMetadata(const char* layer, const char* text,
                                       bool defaultValue) {
    auto lmeta = std::make_shared<LayerMetadata>();
    lmeta->layerName = layer;
    auto metadata = std::make_unique<LBoolMetadata>();
    metadata->parametName = AOCE_LAYER_PARAMET_NAME;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    lmeta->metadata = std::move(metadata);
    layerMetadatas.push_back(lmeta);
}

void LayerMetadataManager::addMetadata(const char* layer, const char* text,
                                       const char* defaultValue) {
    auto lmeta = std::make_shared<LayerMetadata>();
    lmeta->layerName = layer;
    auto metadata = std::make_unique<LStringMetadata>();
    metadata->parametName = AOCE_LAYER_PARAMET_NAME;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    lmeta->metadata = std::move(metadata);
    layerMetadatas.push_back(lmeta);
}

void LayerMetadataManager::addMetadata(const char* layer, const char* text,
                                       int32_t defaultValue, int32_t minValue,
                                       int32_t maxVluae) {
    auto lmeta = std::make_shared<LayerMetadata>();
    lmeta->layerName = layer;
    auto metadata = std::make_unique<LIntMetadata>();
    metadata->parametName = AOCE_LAYER_PARAMET_NAME;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    metadata->minValue = minValue;
    metadata->maxVluae = maxVluae;
    lmeta->metadata = std::move(metadata);
    layerMetadatas.push_back(lmeta);
}

void LayerMetadataManager::addMetadata(const char* layer, const char* text,
                                       float defaultValue, float minValue,
                                       float maxVluae) {
    auto lmeta = std::make_shared<LayerMetadata>();
    lmeta->layerName = layer;
    auto metadata = std::make_unique<LFloatMetadata>();
    metadata->parametName = AOCE_LAYER_PARAMET_NAME;
    metadata->text = text;
    metadata->defaultValue = defaultValue;
    metadata->minValue = minValue;
    metadata->maxVluae = maxVluae;
    lmeta->metadata = std::move(metadata);
    layerMetadatas.push_back(lmeta);
}

BGroupMetadata* LayerMetadataManager::addGroupMetadata(const char* layer,
                                                       const char* parametClass,
                                                       const char* text) {
    auto lmeta = std::make_shared<LayerMetadata>();
    lmeta->layerName = layer;
    auto metadata = std::make_unique<BGroupMetadata>(parametClass);
    metadata->parametName = AOCE_LAYER_PARAMET_NAME;
    metadata->text = text;
    lmeta->metadata = std::move(metadata);
    layerMetadatas.push_back(lmeta);
    return (BGroupMetadata*)lmeta->metadata.get();
}

ILMetadata* LayerMetadataManager::getMetadata(const char* layerName) {
    for (auto& lm : layerMetadatas) {
        if (strcmp(layerName, lm->layerName) == 0) {
            return lm->metadata.get();
        }
    }
    return nullptr;
}

void loadLayerMetadata() {
    auto& lm = LayerMetadataManager::Get();
    BGroupMetadata* bg =
        lm.addGroupMetadata("FlipLayer", "FlipParamet", "FlipParamet");
    bg->addMetadata("bFlipX", "FlipX", false);
    bg->addMetadata("bFlipY", "FlipY", false);
}

ILMetadata* getLayerMetadata(const char* layerName) {
    return LayerMetadataManager::Get().getMetadata(layerName);
}

}  // namespace aoce