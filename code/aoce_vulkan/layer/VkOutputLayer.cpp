#include "VkOutputLayer.hpp"

#include "../vulkan/VulkanManager.hpp"
#include "VkPipeGraph.hpp"
#if WIN32
using namespace aoce::win;
#endif

namespace aoce {
namespace vulkan {
namespace layer {

VkOutputLayer::VkOutputLayer(/* args */) { bOutput = true; }

VkOutputLayer::~VkOutputLayer() {}

void VkOutputLayer::onInitGraph() {
#if WIN32
    winImage = std::make_unique<VkWinImage>();
    bWinInterop = false;
#elif __ANDROID_API__ >= 26
    if (VulkanManager::Get().bInterpGLES) {
        hardwareImage = std::make_unique<HardwareImage>();
    }
#endif
}

void VkOutputLayer::onUpdateParamet() {
    if (paramet.bGpu != oldParamet.bGpu) {
        resetGraph();
    }
}

void VkOutputLayer::onInitVkBuffer() {
    int32_t size = inFormats[0].width * inFormats[0].height *
                   getImageTypeSize(inFormats[0].imageType);
    assert(size > 0);
    // CPU输出
    outBuffer = std::make_unique<VulkanBuffer>();
    outBuffer->initResoure(BufferUsage::store, size,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    cpuData.resize(size);
    if (outFormat.height == 0 || outFormat.width == 0) {
        outFormat = inFormats[0];
    }
#if WIN32
    if (paramet.bGpu && bWinInterop) {
        winImage->bindDx11(vkPipeGraph->getD3D11Device(), outFormats[0]);
    }
#endif
#if __ANDROID_API__ >= 26
    if (VulkanManager::Get().bInterpGLES && paramet.bGpu) {
        hardwareImage->createAndroidBuffer(outFormat);
    }
#endif
    onFormatChanged(inFormats[0], 0);
}

bool VkOutputLayer::onFrame() {
    if (paramet.bCpu) {
        onImageProcessHandle(outBuffer->getCpuData(), inFormats[0], 0);
    }
    if (paramet.bGpu) {
#if WIN32
        winImage->vkCopyTemp(vkPipeGraph->getD3D11Device());
#endif
    }
    return true;
}

void VkOutputLayer::onCommand() {
    if (paramet.bCpu) {
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        context->imageToBuffer(cmd, inTexs[0].get(), outBuffer.get());
    }

    if (paramet.bGpu) {
        VkImage destImage = VK_NULL_HANDLE;
        bool bInterop = false;
#if WIN32
        bInterop = bWinInterop && winImage->getInit();
        destImage = winImage->getImage();
#endif
#if __ANDROID_API__ >= 26
        bInterop = VulkanManager::Get().bInterpGLES;
        destImage = hardwareImage->getImage();
#endif
        if (bInterop && destImage) {
            inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_ACCESS_SHADER_READ_BIT);
            changeLayout(cmd, destImage, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT);
#if WIN32
            VulkanManager::copyImage(cmd, inTexs[0].get(),
                                     winImage->getImage());
#endif
#if __ANDROID_API__ >= 26
            VulkanManager::blitFillImage(cmd, inTexs[0].get(),
                                         hardwareImage->getImage(),
                                         hardwareImage->getFormat().width,
                                         hardwareImage->getFormat().height);

#endif
        }
    }
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
        // inTexs[0]->addBarrier(copyCmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        //                       VK_PIPELINE_STAGE_TRANSFER_BIT);
        int32_t width = outVkTex.width;
        int32_t height = outVkTex.height;
        if (width == 0 && height == 0) {
            width = inFormats[0].width;
            height = inFormats[0].height;
        }
        VulkanManager::blitFillImage(copyCmd, inTexs[0].get(), copyImage, width,
                                     height);
    }
}

#if __ANDROID__
void VkOutputLayer::outGLGpuTex(const GLOutGpuTex& outTex, uint32_t texType,
                                int32_t outIndex) {
    if (!vkPipeGraph->resourceReady()) {
        return;
    }
    int bindType = GL_TEXTURE_2D;
    if (texType > 0) {
        bindType = texType;
    }
    if (paramet.bCpu && !paramet.bGpu && !observer) {
        if (outBuffer && outBuffer->getCpuData()) {
            glBindTexture(bindType, 0);
            // glActiveTexture(GL_TEXTURE0);
            glBindTexture(bindType, outTex.image);
            glTexSubImage2D(bindType, 0, 0, 0, inFormats[0].width,
                            inFormats[0].height, GL_RGBA, GL_UNSIGNED_BYTE,
                            outBuffer->getCpuData());
            glBindTexture(bindType, 0);
        }
    }
#if __ANDROID_API__ >= 26
    if (VulkanManager::Get().bInterpGLES && paramet.bGpu) {
        ImageFormat format = hardwareImage->getFormat();
        int32_t width = outTex.width;
        int32_t height = outTex.height;
        if (width == 0 || height == 0) {
            width = inFormats[0].width;
            height = inFormats[0].height;
        }
        if (format.width != width || format.height != height) {
            format.width = width;
            format.height = height;
            format.imageType = ImageType::rgba8;
            // hardwareImage->createAndroidBuffer(format);
            outFormat = format;
            // 重新生成opengles资源与cmdbuffer
            resetGraph();
            return;
        }
//       int32_t oldIndex = hardwareImage->getTextureId();
//       if (oldIndex < 0 || oldIndex != outTex.image) {
//           hardwareImage->bindGL(outTex.image, bindType);
//       }
       hardwareImage->bindGL(outTex.image, bindType);
    }
#endif
}
#endif

#if WIN32
void VkOutputLayer::outDx11GpuTex(void* device, void* tex) {
    if (!pipeGraph) {
        return;
    }
    if (!bWinInterop) {
        bWinInterop = true;
        resetGraph();
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
        winImage->tempCopyDx11(dxdevice, dxtexture);
    }
}
#endif
}  // namespace layer
}  // namespace vulkan
}  // namespace aoce