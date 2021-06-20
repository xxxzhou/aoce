#include "VkInputLayer.hpp"

#include "../vulkan/VulkanManager.hpp"
#include "VkPipeGraph.hpp"
#if WIN32
using namespace aoce::win;
#endif
namespace aoce {
namespace vulkan {
namespace layer {

VkInputLayer::VkInputLayer(/* args */) {
    bInput = true;
    // setUBOSize(12);
}

VkInputLayer::~VkInputLayer() {}

void VkInputLayer::onDataReady() { bDateUpdate = true; }

void VkInputLayer::onInitGraph() {
#if WIN32
    winImage = std::make_unique<VkWinImage>();
#elif __ANDROID_API__ >= 26
    if (VulkanManager::Get().bInterpGLES) {
        hardwareImage = std::make_unique<HardwareImage>();
    }
#endif
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkInputLayer::onInitVkBuffer() {
    bUsePipe = videoType == VideoType::rgb8 || videoType == VideoType::bgra8 ||
               videoType == VideoType::argb8;
    sizeY = 1;
    int32_t size = inFormats[0].width * inFormats[0].height *
                   getImageTypeSize(inFormats[0].imageType);
    if (bUsePipe) {
        std::string path = "";
        if (videoType == VideoType::rgb8) {
            path = "glsl/inputRGB.comp.spv";
        } else if (videoType == VideoType::bgra8) {
            path = "glsl/inputBRGA.comp.spv";
        } else if (videoType == VideoType::argb8) {
            path = "glsl/inputARGB.comp.spv";
        }
        shader->loadShaderModule(context->device, path);
        assert(shader->shaderStage.module != VK_NULL_HANDLE);
        int imageSize = inFormats[0].width * inFormats[0].height;
        // 如果是rgb-rgba,则先buffer转cs buffer,然后cs shader转rgba.
        // 不直接在cs shader用buf->tex,兼容性考虑cpu map/cs read权限.
        if (videoType == VideoType::rgb8) {
            // 每个线程组处理240个数据,一个线程拿buffer三个数据生成四个点
            sizeX = divUp(imageSize / 4, 240);
            inBufferX = std::make_unique<VulkanBuffer>();
            inBufferX->initResoure(BufferUsage::program, imageSize * 3,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        } else if (videoType == VideoType::argb8 ||
                   videoType == VideoType::bgra8) {
            sizeX = divUp(imageSize, 240);
            inBufferX = std::make_unique<VulkanBuffer>();
            inBufferX->initResoure(BufferUsage::program, imageSize * 4,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }
        size = inBufferX->getBufferSize();
    }
    assert(size > 0);
    inBuffer = std::make_unique<VulkanBuffer>();
    inBuffer->initResoure(BufferUsage::store, size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, frameData);
#if WIN32
    if (paramet.bGpu) {
        winImage->bindDx11(vkPipeGraph->getD3D11Device(), inFormats[0]);        
    }
#endif
#if __ANDROID_API__ >= 26
    if (VulkanManager::Get().bInterpGLES && paramet.bGpu) {
        hardwareImage->createAndroidBuffer(outFormats[0]);
    }
#endif
}

void VkInputLayer::onInitPipe() {
    if (bUsePipe) {
        outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        layout->updateSetLayout(0, 0, &inBufferX->descInfo,
                                &outTexs[0]->descInfo);
        auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
            layout->pipelineLayout, shader->shaderStage);
        VK_CHECK_RESULT(vkCreateComputePipelines(
            context->device, context->pipelineCache, 1, &computePipelineInfo,
            nullptr, &computerPipeline));
    }
}

void VkInputLayer::onCommand() {
    // 不需要CS处理
    if (!bUsePipe) {
        outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (paramet.bCpu) {
            context->bufferToImage(cmd, inBuffer.get(), outTexs[0].get());
        }
        if (paramet.bGpu) {
#if WIN32
            VulkanManager::copyImage(cmd, winImage->getImage(),
                                     outTexs[0]->image, outFormats[0].width,
                                     outFormats[0].height);
#endif
        }
    } else {
        if (paramet.bCpu) {
            VkBufferCopy copyRegion = {};
            copyRegion.size = inBufferX->descInfo.range;
            vkCmdCopyBuffer(cmd, inBuffer->buffer, inBufferX->buffer, 1,
                            &copyRegion);
        }
        if (paramet.bGpu) {
#if WIN32
            context->imageToBuffer(cmd, winImage->getImage(), inBufferX->buffer,
                                   outFormats[0].width, outFormats[0].height);
#endif
        }
        outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computerPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout->pipelineLayout, 0, 1,
                                layout->descSets[0].data(), 0, 0);
        vkCmdDispatch(cmd, sizeX, sizeY, 1);
    }
}

void VkInputLayer::onPreFrame() {
#if WIN32
    if (!paramet.bGpu) {
        return;
    }
    winImage->tempCopyVk(vkPipeGraph->getD3D11Device());
#endif
}

bool VkInputLayer::onFrame() {
    if (inBuffer) {
        if (bDateUpdate) {
            inBuffer->upload(frameData);
            bDateUpdate = false;
        }
        return true;
    }
    return false;
}

void VkInputLayer::inputGpuData(void* device, void* tex) {
#if WIN32
    if (!paramet.bGpu || !pipeGraph) {
        return;
    }
    // 管线在生成资源中
    if (!vkPipeGraph->resourceReady()) {
        return;
    }
    ID3D11Device* dxdevice = (ID3D11Device*)device;
    ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)tex;
    if (dxdevice == nullptr || dxtexture == nullptr) {
        return;
    }
    // 把DX11共享资源复制到另一线程上的device的上纹理
    if (winImage && winImage->getInit()) {
        winImage->dx11CopyTemp(dxdevice, dxtexture);
    }
#endif
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce