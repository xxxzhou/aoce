#include "VkExtraExport.hpp"

#include "layer/VkLinearFilterLayer.hpp"
#include "layer/VkChromKeyLayer.hpp"

using namespace aoce::vulkan::layer;

namespace aoce {
namespace vulkan {

ITLayer<FilterParamet>* createBoxFilterLayer(){
    VkLinearFilterLayer* lineFilter = new VkLinearFilterLayer();
    return lineFilter;
}

ITLayer<ChromKeyParamet>* createChromKeyLayer(){
    VkChromKeyLayer* chromKeyLayer = new VkChromKeyLayer();
    return chromKeyLayer;
}

}
}