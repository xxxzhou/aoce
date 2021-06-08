#pragma once

#include <vector>

#include "../Aoce.hpp"

namespace aoce {

class BMetaData : virtual public ILMetadata {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   protected:
    std::string text = "";
    std::string parametName = "";

   public:
    BMetaData(/* args */){};
    virtual ~BMetaData(){};

    virtual const char* getText() override { return text.c_str(); }
    virtual const char* getParametName() override {
        return parametName.c_str();
    };
    virtual LayerMetadataType getLayerType() override {
        return LayerMetadataType::other;
    }
};

template <typename T>
class BTMetaData : public BMetaData,virtual public ILTMetadata<T> {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   protected:
    T defaultValue = {};

   protected:
    template <typename T1>
    LayerMetadataType getMetaType() {
        return LayerMetadataType::other;
    };

    template <>
    LayerMetadataType getMetaType<const char*>() {
        return LayerMetadataType::astring;
    };

    template <>
    LayerMetadataType getMetaType<bool>() {
        return LayerMetadataType::abool;
    };

    // 特化
    template <>
    LayerMetadataType getMetaType<int32_t>() {
        return LayerMetadataType::aint;
    };

    // 特化
    template <>
    LayerMetadataType getMetaType<float>() {
        return LayerMetadataType::afloat;
    };

   public:
    BTMetaData(/* args */){};
    virtual ~BTMetaData(){};

   public:
    virtual T getDefaultVaule() override { return defaultValue; }
    virtual LayerMetadataType getLayerType() override {
        return getMetaType<T>();
    }
};

template <typename T>
class BTRangeMetaData : public BTMetaData<T>,virtual public ILTRangeMetadata<T> {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   protected:
    T minValue = {};
    T maxVluae = {};

   public:
    BTRangeMetaData(/* args */){};
    virtual ~BTRangeMetaData(){};

    virtual T getMinValue() override { return minValue; }
    virtual T getMaxValue() override { return maxVluae; }
};

class ACOE_EXPORT BGroupMetadata : public BMetaData, public ILGroupMetadata {
    friend class LayerMetadataManager;
    friend class BGroupMetadata;

   private:
    /* data */
    std::vector<std::shared_ptr<BMetaData>> metadatas;
    std::string parametClass = "";

   public:
    BGroupMetadata(const char* paramet_class);
    ~BGroupMetadata();

    virtual int32_t getCount() override;
    virtual ILMetadata* getLMetadata(int32_t index) override;
    virtual const char* getParametClass() override {
        return parametClass.c_str();
    };
    virtual LayerMetadataType getLayerType() override {
        return LayerMetadataType::agroup;
    }

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
    std::unique_ptr<BMetaData> metadata = nullptr;

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
