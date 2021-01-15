#include "VkExtraExport.hpp"

#include "layer/VkLinearFilterLayer.hpp"

using namespace aoce::vulkan::layer;

namespace aoce {
namespace vulkan {

ITLayer<FilterParamet>* createBoxFilterLayer(){
    VkLinearFilterLayer* lineFilter = new VkLinearFilterLayer();
    return lineFilter;
}

}
}