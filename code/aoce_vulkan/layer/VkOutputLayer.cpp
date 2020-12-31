#include "VkOutputLayer.hpp"

#include "../vulkan/VulkanManager.hpp"
#include "VkPipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkOutputLayer::VkOutputLayer(/* args */) { bOutput = true; }

VkOutputLayer::~VkOutputLayer() {
    if (outEvent) {
        vkDestroyEvent(context->device, outEvent, VK_NULL_HANDLE);
        outEvent = VK_NULL_HANDLE;
    }
}

void VkOutputLayer::onInitGraph() {
    VkEventCreateInfo eventInfo = {};
    eventInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventInfo.pNext = nullptr;
    eventInfo.flags = 0;
    vkCreateEvent(context->device, &eventInfo, nullptr, &outEvent);
    vkSetEvent(context->device, outEvent);

#if __ANDROID__
    hardwareImage = std::make_unique<HardwareImage>();
    // hardwareImage->createAndroidBuffer(format);
#endif
}

void VkOutputLayer::onInitVkBuffer() {
    int32_t size = outFormats[0].width * outFormats[0].height *
                   getImageTypeSize(outFormats[0].imageType);
    assert(size > 0);
    // CPU输出
    outBuffer = std::make_unique<VulkanBuffer>();
    outBuffer->initResoure(BufferUsage::store, size,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    cpuData.resize(size);
    // GPU输出
    // const ImageFormat& format = outFormats[0];
    // VkFormat vkft = ImageFormat2Vk(format.imageType);
    // outTex = std::make_unique<VulkanTexture>();
    // outTex->InitResource(outFormats[0].width, outFormats[0].height, vkft,
    //                      VK_IMAGE_USAGE_STORAGE_BIT |
    //                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
    //                          VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    //                      0);
}

bool VkOutputLayer::onFrame() {
    if (paramet.bCpu) {
        onImageProcessHandle(outBuffer->getCpuData(), outFormats[0].width,
                             outFormats[0].height, 0);
        // outBuffer->download(cpuData.data());
    }
    return true;
}

void VkOutputLayer::onPreCmd() {
    if (paramet.bCpu) {
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        context->imageToBuffer(cmd, inTexs[0].get(), outBuffer.get());
    }
    if (paramet.bGpu && outTex.image) {
        // vkCmdWaitEvents(cmd, 1, &outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT,
        //                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, 0,
        //                 nullptr, 0, nullptr);
        // vkCmdResetEvent(cmd, outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT);
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        VkImage outImage = (VkImage)outTex.image;
        changeLayout(cmd, outImage, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
        VulkanManager::blitFillImage(cmd, inTexs[0].get(), outImage,
                                     outTex.width, outTex.height);
        // vkCmdSetEvent(cmd, outEvent, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
#if __ANDROID_API__ >= 26
    if (paramet.bGpu && hardwareImage->getImage()) {
        changeLayout(cmd, hardwareImage->getImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VulkanManager::blitFillImage(cmd, inTexs[0].get(),
                                     hardwareImage->getImage(),
                                     hardwareImage->getFormat().width,
                                     hardwareImage->getFormat().height);
    }
#endif
}

void VkOutputLayer::outGpuTex(const VkOutGpuTex& outVkTex, int32_t outIndex) {
    // outVkTex.commandbuffer外面执行
    if (outVkTex.commandbuffer != nullptr) {
        if (!inTexs[0] || !inTexs[0]->image) {
            return;
        }
        // GPU输出
        VkCommandBuffer copyCmd = (VkCommandBuffer)outVkTex.commandbuffer;
        VkImage copyImage = (VkImage)outVkTex.image;
        // auto res = vkGetEventStatus(context->device, outEvent);
        // assert(res != VK_EVENT_RESET);
        inTexs[0]->addBarrier(copyCmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT);
        VulkanManager::blitFillImage(copyCmd, inTexs[0].get(), copyImage,
                                     outVkTex.width, outVkTex.height);
    } else {
        outTex = outVkTex;
        // 重新生成cmdbuffer
        this->getGraph()->reset();
    }
}

#if __ANDROID__
void VkOutputLayer::outGLGpuTex(const VkOutGpuTex& outTex, uint32_t texType,
                                int32_t outIndex) {
    int bindType = GL_TEXTURE_2D;
    if (texType > 0) {
        bindType = texType;
    }
    if (paramet.bCpu && !paramet.bGpu) {
        if (outBuffer) {
            // 20/12/11,ue4只能这样更新,奇怪了。。。
            if (outTex.commandbuffer) {
                uint8_t* tempPtr = (uint8_t*)outTex.commandbuffer;
                outBuffer->download(tempPtr);
                // memcpy(outTex.commandbuffer,cpuData.data(),cpuData.size());
            } else if(outBuffer->getCpuData()){
                // outBuffer->download(cpuData.data());
                // UE4(Unbind different texture target on the same stage, to
                // avoid OpenGL keeping its data, and potential driver
                // problems.)
                glBindTexture(bindType, 0);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(bindType, outTex.image);
                // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexSubImage2D(bindType, 0, 0, 0, outFormats[0].width,
                                outFormats[0].height, GL_RGBA, GL_UNSIGNED_BYTE,
                                outBuffer->getCpuData());
                // glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                // glTexParameteri(bindType, GL_TEXTURE_MIN_FILTER,
                // GL_LINEAR); glTexParameteri(bindType,
                // GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glBindTexture(bindType, 0);
            }
        }
    }
    if (paramet.bGpu) {
        ImageFormat format = hardwareImage->getFormat();
        uint32_t oldIndex = hardwareImage->getTextureId();
        if (format.width != outTex.width || format.height != outTex.height ||
            oldIndex != outTex.image) {
            format.width = outTex.width;
            format.height = outTex.height;
            format.imageType == ImageType::rgba8;
            hardwareImage->createAndroidBuffer(format);
            // hardwareImage->bindGL(outTex.image, bindType);
            // 重新生成cmdbuffer
            this->getGraph()->reset();
        }
        hardwareImage->bindGL(outTex.image, bindType);        
    }
}
#endif

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce