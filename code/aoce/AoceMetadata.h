#pragma once
#include "Aoce.h"

namespace aoce {

class ILMetadata {
   public:
    virtual ~ILMetadata(){};

    virtual const char* getText() = 0;
    virtual const char* getParametName() = 0;
    virtual LayerMetadataType getLayerType() = 0;
};

template <typename T>
class ILTMetadata : virtual public ILMetadata {
   public:
    virtual ~ILTMetadata(){};
    virtual T getDefaultVaule() = 0;
};

class ILGroupMetadata : virtual public ILMetadata {
   public:
    virtual ~ILGroupMetadata(){};

    virtual int32_t getCount() = 0;
    virtual ILMetadata* getLMetadata(int32_t index) = 0;

    virtual const char* getParametClass() = 0;
};

template <typename T>
class ILTRangeMetadata : virtual public ILTMetadata<T> {
   public:
    virtual ~ILTRangeMetadata(){};

    virtual T getMinValue() = 0;
    virtual T getMaxValue() = 0;
};

}  // namespace aoce