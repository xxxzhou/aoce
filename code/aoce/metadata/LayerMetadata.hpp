#pragma once

#include <vector>

#include "../Aoce.hpp"

namespace aoce {

template <typename T>
inline LayerMetadataType getMetaType() {
    return LayerMetadataType::other;
};

// 特化
template <>
inline LayerMetadataType getMetaType<const char*>() {
    return LayerMetadataType::astring;
};

// 特化
template <>
inline LayerMetadataType getMetaType<bool>() {
    return LayerMetadataType::abool;
};

// 特化
template <>
inline LayerMetadataType getMetaType<int32_t>() {
    return LayerMetadataType::aint;
};

// 特化
template <>
inline LayerMetadataType getMetaType<float>() {
    return LayerMetadataType::afloat;
};

class BMetadata {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   protected:
    std::string text = "";
    std::string parametName = "";

   public:
    BMetadata(/* args */){};
    virtual ~BMetadata(){};
};

template <typename T>
class BTMetaData : public BMetadata, public ILTMetadata<T> {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   protected:
    T defaultValue = {};

   public:
    BTMetaData(/* args */){};
    virtual ~BTMetaData(){};

   public:
    virtual const char* getText() override { return text.c_str(); }
    virtual const char* getParametName() override {
        return parametName.c_str();
    };
    virtual T getDefaultVaule() override { return defaultValue; }
    virtual LayerMetadataType getLayerType() override {
        return getMetaType<T>();
    }
};

template <typename T>
class BTRangeMetaData : public BMetadata, public ILTRangeMetadata<T> {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   protected:
    T defaultValue = {};
    T minValue = {};
    T maxVluae = {};

   public:
    BTRangeMetaData(/* args */){};
    virtual ~BTRangeMetaData(){};

    virtual const char* getText() override { return text.c_str(); }
    virtual const char* getParametName() override {
        return parametName.c_str();
    };
    virtual T getDefaultVaule() override { return defaultValue; }
    virtual LayerMetadataType getLayerType() override {
        return getMetaType<T>();
    }

    virtual T getMinValue() override { return minValue; }
    virtual T getMaxValue() override { return maxVluae; }
};

class ACOE_EXPORT BGroupMetadata : public BMetadata, public ILGroupMetadata {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   private:
    /* data */
    std::vector<std::shared_ptr<ILMetadata>> metadatas;
    std::string parametClass = "";

   public:
    BGroupMetadata(const char* paramet_class);
    ~BGroupMetadata();

    virtual const char* getText() override { return text.c_str(); }
    virtual const char* getParametName() override {
        return parametName.c_str();
    };
    virtual LayerMetadataType getLayerType() override {
        return LayerMetadataType::agroup;
    }
    virtual int32_t getCount() override;
    virtual ILMetadata* getLMetadata(int32_t index) override;
    virtual const char* getParametClass() override {
        return parametClass.c_str();
    };

   public:
    void addMetadata(const char* parametName, const char* text,
                     bool defaultValue);
    void addMetadata(const char* parametName, const char* text,
                     const char* defaultValue);
    void addMetadata(const char* parametName, const char* text,
                     int32_t defaultValue, int32_t minValue, int32_t maxVluae);
    void addMetadata(const char* parametName, const char* text,
                     float defaultValue, float minValue = 0.0f,
                     float maxVluae = 1.0f);
    BGroupMetadata* addGroupMetadata(const char* parametName,
                                     const char* parametClass,
                                     const char* text);
};

typedef BTMetaData<bool> LBoolMetadata;
typedef BTMetaData<const char*> LStringMetadata;

typedef BTRangeMetaData<int32_t> LIntMetadata;
typedef BTRangeMetaData<float> LFloatMetadata;

class LayerMetadata {
    friend class LayerMetadataManager;

   private:
    const char* layerName = "";
    std::unique_ptr<ILMetadata> metadata = nullptr;

   public:
    LayerMetadata(/* args */);
    ~LayerMetadata();
};

class ACOE_EXPORT LayerMetadataManager {
   public:
    static LayerMetadataManager& Get();
    static void clean();
    ~LayerMetadataManager();

   private:
    LayerMetadataManager(/* args */);
    static LayerMetadataManager* instance;

   private:
    std::vector<std::shared_ptr<LayerMetadata>> layerMetadatas;

   public:
    void addMetadata(const char* layer, const char* text, bool defaultValue);
    void addMetadata(const char* layer, const char* text,
                     const char* defaultValue);
    void addMetadata(const char* layer, const char* text, int32_t defaultValue,
                     int32_t minValue, int32_t maxVluae);
    void addMetadata(const char* layer, const char* text, float defaultValue,
                     float minValue = 0.0f, float maxVluae = 1.0f);
    BGroupMetadata* addGroupMetadata(const char* layer,
                                     const char* parametClass,
                                     const char* text);

    ILMetadata* getMetadata(const char* layerName);
};

void loadLayerMetadata();

}  // namespace aoce
