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
    paramet.intensity = 1.0f;
    setUBOSize(sizeof(paramet));
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

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce