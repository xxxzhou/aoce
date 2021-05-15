#include "VkColorMatrixLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkColorMatrixLayer::VkColorMatrixLayer(/* args */) {
    glslPath = "glsl/colorMatrix.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkColorMatrixLayer::~VkColorMatrixLayer() {}

VkHSBLayer::VkHSBLayer(/* args */) {
    glslPath = "glsl/colorMatrix.comp.spv";
    setUBOSize(sizeof(paramet));
    paramet.intensity = 1.0f;
    updateUBO(&paramet);
}

VkHSBLayer::~VkHSBLayer() {}

void VkHSBLayer::parametTransform() {
    updateUBO(&paramet);
    bParametChange = true;
}

void VkHSBLayer::reset() {
    identMat(paramet.mat);
    parametTransform();
}

void VkHSBLayer::rotateHue(const float& h) {
    paramet.mat = huerotateMat(paramet.mat, h);
    parametTransform();
}

void VkHSBLayer::adjustSaturation(const float& s) {
    paramet.mat = saturateMat(paramet.mat, s);
    parametTransform();
}

void VkHSBLayer::adjustBrightness(const float& b) {
    paramet.mat = scaleMat(paramet.mat, {b, b, b});
    parametTransform();
}

VkSepiaLayer::VkSepiaLayer(/* args */) {
    glslPath = "glsl/colorMatrix.comp.spv";
    setUBOSize(sizeof(mparamet));
    mparamet.intensity = 1.0f;
    mparamet.mat = {{0.3588, 0.7044, 0.1368, 0.0},
                    {0.2990, 0.5870, 0.1140, 0.0},
                    {0.2392, 0.4696, 0.0912, 0.0},
                    {0, 0, 0, 1.0}};
    updateUBO(&mparamet);
}

VkSepiaLayer::~VkSepiaLayer() {}

void VkSepiaLayer::onUpdateParamet() {
    if (mparamet.intensity != paramet) {
        mparamet.intensity = paramet;
        updateUBO(&mparamet);
        bParametChange = true;
    }
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce