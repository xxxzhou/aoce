#include "VkNcnnInLayer.hpp"

#include "aoce/AoceMath.h"
#include "aoce/layer/PipeGraph.hpp"
#include "aoce_vulkan/vulkan/VulkanPipeline.hpp"
#if __ANDROID__
#include "aoce_vulkan/android/vulkan_wrapper.h"
#endif

namespace aoce {

VkNcnnInLayer::VkNcnnInLayer() {
    bOutput = true;
    paramet.outWidth = 320;
    paramet.outHeight = 240;
    paramet.mean = {0.0f, 0.0f, 0.0f, 0.0f};
    paramet.scale = {1.0f, 1.0f, 1.0f, 1.0f};
    setUBOSize(sizeof(paramet));
    updateUBO(&paramet);
    bOneVkDevice = getNgParamet().bOneVkDevice;
}

VkNcnnInLayer::~VkNcnnInLayer() {}

void VkNcnnInLayer::setObserver(INcnnInLayerObserver* observer,
                                ImageType outType) {
    this->observer = observer;
    outImageType = outType;
}

void VkNcnnInLayer::updateParamet(const NcnnInParamet& nparamet, bool fP16) {
    if (paramet.outHeight != nparamet.outHeight ||
        paramet.outWidth != nparamet.outHeight || bFP16 != fP16) {
        bFP16 = fP16;
        paramet.outWidth = nparamet.outWidth;
        paramet.outHeight = nparamet.outHeight;
        resetGraph();
    }
    if (!(paramet.mean == nparamet.mean) ||
        !(paramet.scale == nparamet.scale)) {
        paramet.mean = nparamet.mean;
        paramet.scale = nparamet.scale;
        onParametChange(true);
    }
}

void VkNcnnInLayer::onParametChange(bool bUpdateUBO) {
    updateUBO(&paramet);
    bParametChange = true;
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

void VkNcnnInLayer::onInitLayer() {
    std::string fileName =
        outImageType == ImageType::rgb8 ? "ncnnInMatRGB" : "ncnnInMatBGR";
    if (bFP16) {
        fileName += "F16";
    }
    glslPath = "glsl/" + fileName + ".comp.spv";
    shader->loadShaderModule(context->device, glslPath);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    sizeX = divUp(paramet.outWidth, 16);
    sizeY = divUp(paramet.outHeight, 16);
}

void VkNcnnInLayer::onInitVkBuffer() {
    // vkmat
    inVkMat.create(paramet.outWidth, paramet.outHeight, 3, bFP16 ? 2u : 4u,
                   getNgParamet().vkAllocator);
    outFormats[0].width = paramet.outWidth;
    outFormats[0].height = paramet.outHeight;
    outFormats[0].imageType = outImageType;
    int32_t bufferSize =
        paramet.outWidth * paramet.outHeight * 3 * (bFP16 ? 2u : 4u);
    if (bufferSize <= 0) {
        logMessage(LogLevel::error, "aoce ncnn layer requires input size. ");
    }
    assert(bufferSize > 0);
    outBuffer = std::make_unique<VulkanBuffer>();
    outBuffer->initResoure(
        BufferUsage::store, bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    // if (!bOneVkDevice) {
    //     inVkMat.mapped().data = outBuffer->getCpuData();
    // }
}

void VkNcnnInLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    inTexs[0]->descInfo.sampler = context->linearSampler;
    if (bOneVkDevice) {
        VkDescriptorBufferInfo vbInfo = {};
        vbInfo.buffer = inVkMat.buffer();
        vbInfo.offset = inVkMat.buffer_offset();
        vbInfo.range = inVkMat.buffer_capacity();
        layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &vbInfo,
                                &constBuf->descInfo);
    } else {
        layout->updateSetLayout(0, 0, &inTexs[0]->descInfo,
                                &outBuffer->descInfo, &constBuf->descInfo);
    }
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

void VkNcnnInLayer::onCommand() {
    VkBuffer vbuffer = bOneVkDevice ? inVkMat.buffer() : outBuffer->buffer;
    inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    changeLayout(cmd, vbuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                 VK_ACCESS_MEMORY_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
    changeLayout(cmd, vbuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_ACCESS_MEMORY_READ_BIT);
}

bool VkNcnnInLayer::onFrame() {
    if (!bOneVkDevice) {
        memcpy(inVkMat.mapped_ptr(), outBuffer->getCpuData(),
               outBuffer->getBufferSize());
        // getNgParamet().vkAllocator->flush(inVkMat.data);
    }
    if (observer) {
        observer->onResult(inVkMat, inFormats[0]);
    }
    return true;
}

VkNcnnInCropLayer::VkNcnnInCropLayer() {
    cropParamet.ncnnIn = paramet;
    setUBOSize(sizeof(cropParamet));
    updateUBO(&cropParamet);
}

VkNcnnInCropLayer::~VkNcnnInCropLayer() {}

void VkNcnnInCropLayer::setObserver(INcnnInLayerObserver* observer,
                                    ImageType outType) {
    this->observer = observer;
    outImageType = outType;
}

void VkNcnnInCropLayer::onParametChange(bool bUpdateUBO) {
    cropParamet.ncnnIn = paramet;
    updateUBO(&cropParamet);
    bParametChange = true;
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

void VkNcnnInCropLayer::onInitLayer() {
    std::string fileName = outImageType == ImageType::rgb8 ? "ncnnInCropMatRGB"
                                                           : "ncnnInCropMatBGR";
    if (bFP16) {
        fileName += "F16";
    }
    glslPath = "glsl/" + fileName + ".comp.spv";
    shader->loadShaderModule(context->device, glslPath);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    sizeX = divUp(paramet.outWidth, 16);
    sizeY = divUp(paramet.outHeight, 16);
}

bool VkNcnnInCropLayer::onFrame() {
    if (!bFindBox) {
        return true;
    }
    return VkNcnnInLayer::onFrame();
}

VkNcnnUploadLayer::VkNcnnUploadLayer() { bInput = true; }

VkNcnnUploadLayer::~VkNcnnUploadLayer() {}

void VkNcnnUploadLayer::setImageFormat(const ImageFormat& inFormat, bool fP16) {
    bFP16 = fP16;
    imageFormat = inFormat;
}

void VkNcnnUploadLayer::uploadBuffer(const void* data) {
    if (inBuffer) {
        inBuffer->upload((uint8_t*)data);
    }
}

VulkanBuffer* VkNcnnUploadLayer::getVkBuffer() { return inBuffer.get(); }

void VkNcnnUploadLayer::onInitGraph() {
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
    glslPath = "glsl/ncnnUpload.comp.spv";
    if (bFP16) {
        glslPath = "glsl/ncnnUploadF16.comp.spv";
    }
    shader->loadShaderModule(context->device, glslPath);
    assert(shader->shaderStage.module != VK_NULL_HANDLE);
    // 输出数据格式
    outFormats[0].width = imageFormat.width;
    outFormats[0].height = imageFormat.height;
    outFormats[0].imageType = ImageType::r8;
    sizeX = divUp(imageFormat.width, groupX);
    sizeY = divUp(imageFormat.height, groupY);
}

void VkNcnnUploadLayer::onInitVkBuffer() {
    int32_t elemtSize = bFP16 ? 2 : 4;
    int32_t bufferSize = imageFormat.width * imageFormat.height * elemtSize;
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