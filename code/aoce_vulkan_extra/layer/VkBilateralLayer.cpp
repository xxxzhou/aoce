#include "VkBilateralLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
VkBilateralLayer::VkBilateralLayer(/* args */) {
    glslPath = "glsl/bilateral.comp.spv";
    setUBOSize(sizeof(paramet));
    transformParamet();
}

VkBilateralLayer::~VkBilateralLayer() {}

void VkBilateralLayer::transformParamet() {
    BilateralParamet tparamet = {};
    tparamet.kernelSize = paramet.kernelSize;
    tparamet.sigma_color = -0.5f / (paramet.sigma_color * paramet.sigma_color);
    tparamet.sigma_spatial =
        -0.5f / (paramet.sigma_spatial * paramet.sigma_spatial);
    updateUBO(&tparamet);
}

void VkBilateralLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    transformParamet();
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce