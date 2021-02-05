#include "VkOutputLayer.hpp"

#include "../vulkan/VulkanManager.hpp"
#include "VkPipeGraph.hpp"

using namespace aoce::win;

namespace aoce {
namespace vulkan {
namespace layer {

VkOutputLayer::VkOutputLayer(/* args */) { bOutput = true; }

VkOutputLayer::~VkOutputLayer() {}

void VkOutputLayer::onInitGraph() {
#if WIN32
    winImage = std::make_unique<VkWinImage>();
#elif __ANDROID__
    hardwareImage = std::make_unique<HardwareImage>();
    // hardwareImage->createAndroidBuffer(format);
#endif
    bWinInterop = false;
}

void VkOutputLayer::onUpdateParamet() {
    if (pipeGraph && paramet.bGpu != oldParamet.bGpu) {
        pipeGraph->reset();
    }
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
#if WIN32
    if (paramet.bGpu && bWinInterop) {
        winImage->bindDx11(vkPipeGraph->getD3D11Device(), outFormats[0]);
        if (winImage->getInit()) {
            vkPipeGraph->addOutMemory(winImage.get());
        }
    }
#endif
}

bool VkOutputLayer::onFrame() {
    if (paramet.bCpu) {
        onImageProcessHandle(outBuffer->getCpuData(), outFormats[0], 0);
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
#if WIN32
    if (paramet.bGpu && bWinInterop && winImage->getInit()) {
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        changeLayout(cmd, winImage->getImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VulkanManager::copyImage(cmd, inTexs[0].get(), winImage->getImage());
    }
#endif
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

void VkOutputLayer::outVkGpuTex(const VkOutGpuTex& outVkTex, int32_t outIndex) {
    if (!vkPipeGraph->resourceReady()) {
        return;
    }
    if (outVkTex.commandbuffer != nullptr) {
        if (inTexs.empty() || !inTexs[0] || !inTexs[0]->image) {
            return;
        }
        // GPU输出
        VkCommandBuffer copyCmd = (VkCommandBuffer)outVkTex.commandbuffer;
        VkImage copyImage = (VkImage)outVkTex.image;
        inTexs[0]->addBarrier(copyCmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT);
        VulkanManager::blitFillImage(copyCmd, inTexs[0].get(), copyImage,
                                     outVkTex.width, outVkTex.height);
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
            } else if (outBuffer->getCpuData()) {
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

#if WIN32
void VkOutputLayer::outDx11GpuTex(void* device, void* tex) {
    if (!bWinInterop) {
        bWinInterop = true;
        vkPipeGraph->reset();
        return;
    }
    if (!vkPipeGraph->resourceReady()) {
        return;
    }

    ID3D11Device* dxdevice = (ID3D11Device*)device;
    ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)tex;
    if (!paramet.bGpu || dxdevice == nullptr || dxtexture == nullptr) {
        return;
    }
    // 把DX11共享资源复制到另一线程上的device的上纹理
    if (paramet.bGpu && winImage->getInit()) {
        // copySharedToTexture(dxdevice, winImage->getHandle(), dxtexture);
        winImage->tempCopyDx11(dxdevice, dxtexture);
    }
}
#endif
}  // namespace layer
}  // namespace vulkan
}  // namespace aoce