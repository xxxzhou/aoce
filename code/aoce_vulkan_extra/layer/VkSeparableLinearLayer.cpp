#include "VkSeparableLinearLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"
#define PATCH_PER_BLOCK 4

namespace aoce {
namespace vulkan {
namespace layer {

VkSeparableLayer::VkSeparableLayer(ImageType imageType) {
    this->imageType = imageType;
    setUBOSize(8);
    glslPath = "glsl/filterRow.comp.spv";
    if (imageType == ImageType::r8) {
        glslPath = "glsl/filterRowC1.comp.spv";
    } else if (imageType == ImageType::rgba32f) {
        glslPath = "glsl/filterRowF4.comp.spv";
    }
}

VkSeparableLayer::~VkSeparableLayer() {}

void VkSeparableLayer::updateBuffer(std::vector<float> data) {
    int32_t size = data.size();
    std::vector<int32_t> ubo = {size, size / 2};
    updateUBO(ubo.data());

    kernelBuffer = std::make_unique<VulkanBuffer>();
    kernelBuffer->initResoure(BufferUsage::onestore, size * sizeof(float),
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              (uint8_t*)data.data());
}

void VkSeparableLayer::onInitGraph() {
    inFormats[0].imageType = imageType;
    outFormats[0].imageType = imageType;

    shader->loadShaderModule(context->device, glslPath);

    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkSeparableLayer::onInitLayer() {
    // row[rSizeX,sizeY] column[sizeX,cSizeY]
    // 线程组只取平常宽的1/4,每个线程处理四个点
    sizeX = divUp(inFormats[0].width, groupX * PATCH_PER_BLOCK);
    sizeY = divUp(inFormats[0].height, groupY);
}

void VkSeparableLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &outTexs[0]->descInfo,
                            &constBuf->descInfo, &kernelBuffer->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

VkSeparableLinearLayer::VkSeparableLinearLayer(ImageType imageType)
    : VkSeparableLayer(imageType) {
    rowLayer = std::make_unique<VkSeparableLayer>(imageType);
    glslPath = "glsl/filterColumn.comp.spv";
    if (imageType == ImageType::r8) {
        glslPath = "glsl/filterColumnC1.comp.spv";
    } else if (imageType == ImageType::rgba32f) {
        glslPath = "glsl/filterColumnF4.comp.spv";
    }
}

VkSeparableLinearLayer::~VkSeparableLinearLayer() {}

void VkSeparableLinearLayer::onInitGraph() {
    VkSeparableLayer::onInitGraph();
    pipeGraph->addNode(rowLayer.get());
}

void VkSeparableLinearLayer::onInitNode() {
    rowLayer->getNode()->addLine(getNode(), 0, 0);
    getNode()->setStartNode(rowLayer->getNode());
}

void VkSeparableLinearLayer::onInitLayer() {
    sizeX = divUp(inFormats[0].width, groupX);
    // 线程组只取平常宽的1/4,每个线程处理四个点
    sizeY = divUp(inFormats[0].height, groupY * PATCH_PER_BLOCK);
}

VkBoxBlurSLayer::VkBoxBlurSLayer(ImageType imageType)
    : VkSeparableLinearLayer(imageType) {
    paramet.kernelSizeX = 5;
    paramet.kernelSizeY = 5;
}

VkBoxBlurSLayer::~VkBoxBlurSLayer() {}

void VkBoxBlurSLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    resetGraph();
}

void VkBoxBlurSLayer::getKernel(int size, std::vector<float>& kernels) {
    kernels.resize(size);
    float sum = 1.0 / size;
    for (int i = 0; i < size; i++) {
        kernels[i] = sum;
    }
}

void VkBoxBlurSLayer::onInitLayer() {
    VkSeparableLinearLayer::onInitLayer();
    int ksizex = paramet.kernelSizeX;
    std::vector<float> karrayX;
    std::vector<float> karrayY;
    getKernel(paramet.kernelSizeX, karrayX);
    getKernel(paramet.kernelSizeY, karrayY);
    rowLayer->updateBuffer(karrayX);
    updateBuffer(karrayY);
}

VkGaussianBlurSLayer::VkGaussianBlurSLayer(ImageType imageType)
    : VkSeparableLinearLayer(imageType) {
}

VkGaussianBlurSLayer::~VkGaussianBlurSLayer() {}

void VkGaussianBlurSLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    resetGraph();
}

void VkGaussianBlurSLayer::onInitLayer() {
    VkSeparableLinearLayer::onInitLayer();
    int ksize = paramet.blurRadius * 2 + 1;   
    if (paramet.sigma <= 0.0f) {
        paramet.sigma = ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8;
    }
    std::vector<float> karray(ksize);
    double sum = 0.0;
    double scale = 1.0f / (paramet.sigma * paramet.sigma * 2.0);
    for (int i = 0; i < ksize; i++) {
        int x = i - (ksize - 1) / 2;
        karray[i] = exp(-scale * (x * x));
        sum += karray[i];
    }
    sum = 1.0 / sum;
    for (int i = 0; i < ksize; i++) {
        karray[i] *= sum;
    }
    rowLayer->updateBuffer(karray);
    updateBuffer(karray);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce