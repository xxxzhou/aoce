#include "VkRGBA2YUVLayer.hpp"

#include "VkPipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkRGBA2YUVLayer::VkRGBA2YUVLayer(/* args */) { setUBOSize(12); }

VkRGBA2YUVLayer::~VkRGBA2YUVLayer() {}

void VkRGBA2YUVLayer::onInitGraph() {
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkRGBA2YUVLayer::onUpdateParamet() {
    assert(getYuvIndex(paramet.type) > 0);
    if (pipeGraph) {
        pipeGraph->reset();
    }
}

void VkRGBA2YUVLayer::onInitLayer() {
    int32_t yuvType = getYuvIndex(paramet.type);
#ifdef AOCE_INSTALL_TALKTO
    if (paramet.type == VideoType::yuy2P && paramet.special != 0) {
        yuvType = 0;
    }
#endif
    assert(yuvType >= 0);
    // nv12/yuv420P/yuy2P
    std::string path = "glsl/rgba2yuvV1.comp.spv";
    if (yuvType > 3) {
        path = "glsl/rgba2yuvV2.comp.spv";
    }
    shader->loadShaderModule(context->device, path);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    // 带P/SP的格式由r8转rgba8
    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
    if (paramet.type == VideoType::nv12 || paramet.type == VideoType::yuv420P) {
        outFormats[0].height = inFormats[0].height * 3 / 2;
        // 一个线程处理四个点
        sizeX = divUp(inFormats[0].width, 2 * groupX);
        sizeY = divUp(inFormats[0].height, 2 * groupY);
    } else if (paramet.type == VideoType::yuy2P) {
        outFormats[0].height = inFormats[0].height * 2;
        // 一个线程处理二个点
        sizeX = divUp(inFormats[0].width, 2 * groupX);
        if (paramet.special == 0) {
            sizeY = divUp(inFormats[0].height, groupY);
        } else {
            sizeY = divUp(inFormats[0].height, 2 * groupY);
        }
    } else if (paramet.type == VideoType::yuv2I ||
               paramet.type == VideoType::yvyuI ||
               paramet.type == VideoType::uyvyI) {
        outFormats[0].imageType = ImageType::rgba8;
        // 一个线程处理二个点,yuyv四点组合成一个元素,和rgba类似
        outFormats[0].width = inFormats[0].width / 2;
        sizeX = divUp(inFormats[0].width, groupX);
        sizeY = divUp(inFormats[0].height, groupY);
    }
    // 更新constBufCpu
    std::vector<int> ubo = {outFormats[0].width, outFormats[0].height, yuvType};
    memcpy(constBufCpu.data(), ubo.data(), conBufSize);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce