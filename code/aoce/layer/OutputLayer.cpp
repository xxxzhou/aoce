#include "OutputLayer.hpp"

namespace aoce {

void OutputLayer::onImageProcessHandle(uint8_t* data,
                                       const ImageFormat& imageFormat,
                                       int32_t outIndex) {
    if (observer) {
        observer->onImageProcess(data, imageFormat, outIndex);
    }
}

void OutputLayer::onFormatChanged(const ImageFormat& imageFormat,
                                  int32_t outIndex) {
    if (observer) {
        observer->onFormatChanged(imageFormat, outIndex);
    }
}

void OutputLayer::setObserver(IOutputLayerObserver* observer) {
    this->observer = observer;
}

}  // namespace aoce