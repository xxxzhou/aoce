#include "VkExtraModule.hpp"

#include "../aoce/metadata/LayerMetadata.hpp"

namespace aoce {
namespace vulkan {

VkExtraModule::VkExtraModule() {}

VkExtraModule::~VkExtraModule() {}

bool VkExtraModule::loadModule() {
    auto& lm = LayerMetadataManager::Get();
    lm.addMetadata("BrightnessLayer", "Brightness", 0.0f, -1.0, 1.0f);
    lm.addMetadata("ExposureLayer", "Exposure", 0.0f, -10.0, 10.0f);
    auto skinMd =
        lm.addGroupMetadata("SkinToneLayer", "SkinToneParamet", "SkinTone");
    skinMd->addMetadata("skinToneAdjust", "skinToneAdjust", 0.0f, -3.0f, 3.0f);
    skinMd->addMetadata("skinHue", "skinHue", 0.0f, 0.0f, 1.0f);
    skinMd->addMetadata("skinHueThreshold", "skinHueThreshold", 40.0f, 0.0f,
                        360.0f);
    skinMd->addMetadata("maxHueShift", "maxHueShift", 0.25f, 0.0f, 1.0f);
    skinMd->addMetadata("maxSaturationShift", "maxSaturationShift", 0.4f, 0.0f,
                        1.0f);
    skinMd->addMetadata("upperSkinToneColor", "upperSkinToneColor", false);
    return true;
}

void VkExtraModule::unloadModule() {}

ADD_MODULE(VkExtraModule, aoce_vulkan_extra)

}  // namespace vulkan
}  // namespace aoce