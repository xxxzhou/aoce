#include "VkCropLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkCropLayer::VkCropLayer(/* args */) {}

VkCropLayer::~VkCropLayer() {}

bool VkCropLayer::parametTransform() {
    paramet.centerX = std::max(std::min(paramet.width, 1.0f), 0.0f);
    paramet.centerX = std::max(std::min(paramet.height, 1.0f), 0.0f);

    float maxWidth = std::min(1.0f - paramet.centerX, paramet.centerX) * 2.0f;
    float maxHeight = std::min(1.0f - paramet.centerY, paramet.centerY) * 2.0f;

    paramet.width = std::max(std::min(paramet.width, maxWidth), 0.0f);
    paramet.height = std::max(std::min(paramet.height, maxHeight), 0.0f);

    float left = inFormats[0].width * (paramet.centerX - paramet.width / 2.0f);
    float top = inFormats[0].height * (paramet.centerY - paramet.height / 2.0f);
    float leftTop[2] = {left, top};
    updateUBO(&leftTop[0]);

    if (paramet.width == oldParamet.width &&
        paramet.height == oldParamet.height) {
        return true;
    }
    return false;
}

void VkCropLayer::onUpdateParamet() {
    if (!parametTransform()) {
        resetGraph();
    }
}

void VkCropLayer::onInitLayer() {
    // 如果paramet里的width/height为0,则只取一个数据,可以给别层运算,移除map cpu
    parametTransform();
    outFormats[0].width = (int32_t)(paramet.width * inFormats[0].width) + 1;
    outFormats[0].height = (int32_t)(paramet.height * inFormats[0].height) + 1;  

    sizeX = divUp(outFormats[0].width, groupX);
    sizeY = divUp(outFormats[0].height, groupY);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce