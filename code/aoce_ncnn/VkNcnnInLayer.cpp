#include "VkNcnnInLayer.hpp"

#include "aoce/AoceMath.h"
#include "aoce/layer/PipeGraph.hpp"
#include "aoce_vulkan/vulkan/VulkanPipeline.hpp"
#if __ANDROID__
#include "aoce_vulkan/android/vulkan_wrapper.h"
#endif

namespace aoce {

DrawProperty::DrawProperty() {}

DrawProperty::~DrawProperty() {}

void DrawProperty::setDraw(bool bDraw) { this->bDraw = bDraw; }

void DrawProperty::setDraw(int32_t radius, const vec4 color) {
    this->radius = std::max(1, radius);
    this->color = color;
}

VkNcnnInLayer::VkNcnnInLayer() {
    bOutput = true;
    glslPath = "glsl/ncnnInMatBGR.comp.spv";
    paramet.outWidth = 320;
    paramet.outHeight = 240;
    paramet.mean = {0.0f, 0.0f, 0.0f, 0.0f};
    paramet.scale = {1.0f, 1.0f, 1.0f, 1.0f};
    setUBOSize(sizeof(paramet));
    updateUBO(&paramet);
}

VkNcnnInLayer::~VkNcnnInLayer() {}

void VkNcnnInLayer::setObserver(INcnnInLayerObserver* observer,
                                ImageType outType) {
    this->observer = observer;
    outImageType = outType;
    if (outImageType == ImageType::rgb8) {
        glslPath = "glsl/ncnnInMatRGB.comp.spv";
    }
}

void VkNcnnInLayer::updateParamet(const NcnnInParamet& nparamet) {
    if (paramet.outHeight != nparamet.outHeight ||
        paramet.outWidth != nparamet.outHeight) {
        paramet.outWidth = nparamet.outWidth;
        paramet.outHeight = nparamet.outHeight;
        resetGraph();
    }
    if (!(paramet.mean == nparamet.mean) ||
        !(paramet.scale == nparamet.scale)) {
        paramet.mean = nparamet.mean;
        paramet.scale = nparamet.scale;
        updateUBO(&paramet);
        bParametChange = true;
    }
}

bool VkNcnnInLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

void VkNcnnInLayer::onInitGraph() {
    if (layout->pipelineLayout != VK_NULL_HANDLE) {
        return;
    }
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkNcnnInLayer::onInitVkBuffer() {
    outFormats[0].width = paramet.outWidth;
    outFormats[0].height = paramet.outHeight;
    outFormats[0].imageType = outImageType;

    shader->loadShaderModule(context->device, glslPath);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);

    int32_t bufferSize = paramet.outWidth * paramet.outHeight * 3 * 4;
    if (bufferSize <= 0) {
        logMessage(LogLevel::error, "aoce ncnn layer requires input size. ");
    }
    assert(bufferSize > 0);
    outBuffer = std::make_unique<VulkanBuffer>();
    outBuffer->initResoure(
        BufferUsage::store, bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);  // |
                                              // VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    //    outBufferX = std::make_unique<VulkanBuffer>();
    //    outBufferX->initResoure(BufferUsage::store, bufferSize,
    //                            VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    sizeX = divUp(paramet.outWidth, 16);
    sizeY = divUp(paramet.outHeight, 16);
}

void VkNcnnInLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    inTexs[0]->descInfo.sampler = context->linearSampler;
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &outBuffer->descInfo,
                            &constBuf->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

void VkNcnnInLayer::onCommand() {
    inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    outBuffer->addBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
    outBuffer->addBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    // 复制CS BUFFER到中转BUFFER上
    // VkBufferCopy copyRegion = {};
    // copyRegion.size = outBuffer->getBufferSize();
    // vkCmdCopyBuffer(cmd, outBuffer->buffer, outBufferX->buffer, 1,
    // &copyRegion);
}

bool VkNcnnInLayer::onFrame() {
    if (observer) {
        observer->onResult(outBuffer.get(), inFormats[0]);
    }
    return true;
}

VkNcnnInCropLayer::VkNcnnInCropLayer() {
    glslPath = "glsl/ncnnInCropMatBGR.comp.spv";
    cropParamet.ncnnIn = paramet;
    setUBOSize(sizeof(cropParamet));
    updateUBO(&cropParamet);
}

VkNcnnInCropLayer::~VkNcnnInCropLayer() {}

void VkNcnnInCropLayer::setObserver(INcnnInLayerObserver* observer,
                                    ImageType outType) {
    this->observer = observer;
    outImageType = outType;
    if (outImageType == ImageType::rgb8) {
        glslPath = "glsl/ncnnInCropMatRGB.comp.spv";
    }
}

void VkNcnnInCropLayer::updateParamet(const NcnnInParamet& nparamet) {
    if (cropParamet.ncnnIn.outHeight != nparamet.outHeight ||
        cropParamet.ncnnIn.outWidth != nparamet.outHeight) {
        cropParamet.ncnnIn.outWidth = nparamet.outWidth;
        cropParamet.ncnnIn.outHeight = nparamet.outHeight;
        resetGraph();
    }
    if (!(cropParamet.ncnnIn.mean == nparamet.mean) ||
        !(cropParamet.ncnnIn.scale == nparamet.scale)) {
        cropParamet.ncnnIn.mean = nparamet.mean;
        cropParamet.ncnnIn.scale = nparamet.scale;
        updateUBO(&cropParamet);
        bParametChange = true;
    }
}

void VkNcnnInCropLayer::getInFaceBox(FaceBox& box) { box = faceBox; }

void VkNcnnInCropLayer::detectFaceBox(const FaceBox* boxs, int32_t lenght) {
    bFindBox = lenght > 0;
    if (bFindBox) {
        const FaceBox& box = boxs[0];
        cropParamet.crop.x = box.x1;
        cropParamet.crop.y = box.x2;
        cropParamet.crop.z = box.y1;
        cropParamet.crop.w = box.y2;
        updateUBO(&cropParamet);
        bParametChange = true;
        // 记录传入的大小
        faceBox = box;
    }
}

bool VkNcnnInCropLayer::onFrame() {
    if (!bFindBox) {
        return true;
    }
    return VkNcnnInLayer::onFrame();
}

VkNcnnUploadLayer::VkNcnnUploadLayer() {
    bInput = true;
    glslPath = "glsl/ncnnUpload.comp.spv";
}

VkNcnnUploadLayer::~VkNcnnUploadLayer() {}

void VkNcnnUploadLayer::setImageFormat(const ImageFormat& inFormat) {
    imageFormat = inFormat;
}

void VkNcnnUploadLayer::uploadBuffer(const void* data) {
    if (inBuffer) {
        inBuffer->upload((uint8_t*)data);
    }
}

void VkNcnnUploadLayer::onInitGraph() {
    // 加载shader
    shader->loadShaderModule(context->device, glslPath);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    if (layout->pipelineLayout != VK_NULL_HANDLE) {
        return;
    }
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkNcnnUploadLayer::onInitLayer() {
    // 输出数据格式
    outFormats[0].width = imageFormat.width;
    outFormats[0].height = imageFormat.height;
    outFormats[0].imageType = ImageType::r8;
    sizeX = divUp(imageFormat.width, groupX);
    sizeY = divUp(imageFormat.width, groupY);
}

void VkNcnnUploadLayer::onInitVkBuffer() {
    int32_t bufferSize = imageFormat.width * imageFormat.height * 4;
    inBuffer = std::make_unique<VulkanBuffer>();
    inBuffer->initResoure(BufferUsage::store, bufferSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void VkNcnnUploadLayer::onInitPipe() {
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->updateSetLayout(0, 0, &inBuffer->descInfo, &outTexs[0]->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

void VkNcnnUploadLayer::onCommand() {
    outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
}

}  // namespace aoce