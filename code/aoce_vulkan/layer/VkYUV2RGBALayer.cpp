#include "VkYUV2RGBALayer.hpp"

#include "VkPipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkYUV2RGBALayer::VkYUV2RGBALayer(/* args */) { setUBOSize(12); }

VkYUV2RGBALayer::~VkYUV2RGBALayer() {}

void VkYUV2RGBALayer::onInitGraph() {
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkYUV2RGBALayer::onUpdateParamet() {
    assert(getYuvIndex(paramet.type) > 0);
    pipeGraph->reset();
}

void VkYUV2RGBALayer::onInitLayer() {
    int32_t yuvType = getYuvIndex(paramet.type);
    assert(yuvType > 0);
    // nv12/yuv420P/yuy2P
    std::string path = "glsl/yuv2rgbaV1.comp.spv";
    if (yuvType > 3) {
        path = "glsl/yuv2rgbaV2.comp.spv";
    }
    shader->loadShaderModule(context->device, path);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    // 带P/SP的格式由r8转rgba8
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::rgba8;
    if (paramet.type == VideoType::nv12 || paramet.type == VideoType::yuv420P) {
        outFormats[0].height = inFormats[0].height * 2 / 3;
        // 一个线程处理四个点
        sizeX = divUp(outFormats[0].width, 2 * groupX);
        sizeY = divUp(outFormats[0].height, 2 * groupY);
    } else if (paramet.type == VideoType::yuy2P) {
        outFormats[0].height = inFormats[0].height / 2;
        // 一个线程处理二个点
        sizeX = divUp(outFormats[0].width, 2 * groupX);
        sizeY = divUp(outFormats[0].height, groupY);
    } else if (paramet.type == VideoType::yuv2I ||
               paramet.type == VideoType::yvyuI ||
               paramet.type == VideoType::uyvyI) {
        inFormats[0].imageType = ImageType::rgba8;
        // 一个线程处理二个点,yuyv四点组合成一个元素,和rgba类似
        outFormats[0].width = inFormats[0].width * 2;
        sizeX = divUp(inFormats[0].width, groupX);
        sizeY = divUp(inFormats[0].height, groupY);
    }
    // 更新constBufCpu
    std::vector<int> ubo = {outFormats[0].width, outFormats[0].height,
                            getYuvIndex(paramet.type)};
    memcpy(constBufCpu.data(), ubo.data(), conBufSize);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce