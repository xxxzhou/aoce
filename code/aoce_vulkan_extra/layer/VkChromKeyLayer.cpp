#include "VkChromKeyLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkChromKeyLayer::VkChromKeyLayer(/* args */) {
    setUBOSize(sizeof(ChromKeyParamet), true);
    glslPath = "glsl/chromaKey.comp.spv";
    onUpdateParamet();    
}

VkChromKeyLayer::~VkChromKeyLayer() {} 

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce