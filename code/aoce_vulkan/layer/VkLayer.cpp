#include "VkLayer.hpp"

#include "../vulkan/VulkanPipeline.hpp"
#include "VkPipeGraph.hpp"

#ifdef _WIN32
#include <windows.h>
#elif __ANDROID__
#include <android/asset_manager.h>
#endif

namespace aoce {
namespace vulkan {
namespace layer {

VkLayer::VkLayer(/* args */) { gpu = GpuType::vulkan; }

VkLayer::~VkLayer() {}

void VkLayer::onInit() {
    vkPipeGraph =
        static_cast<VkPipeGraph*>(pipeGraph);  // dynamic_cast android open rtti
    context = vkPipeGraph->getContext();
    assert(context != nullptr);
    cmd = context->computerCmd;
    if (!bInput) {
        inTexs.resize(inCount);
    }
    outTexs.resize(outCount);
    layout = std::make_unique<UBOLayout>(context);
    shader = std::make_unique<VulkanShader>();
    onInitGraph();
}

// void VkLayer::onInitLayer() { }

void VkLayer::onInitBuffer() {
    if (!bInput) {
        inTexs.clear();
        for (int32_t i = 0; i < inCount; i++) {
            auto& inLayer = this->inLayers[i];
            inTexs.push_back(
                vkPipeGraph->getOutTex(inLayer.nodeIndex, inLayer.outputIndex));
        }
    }
    if (!bOutput) {
        outTexs.clear();
        for (int32_t i = 0; i < outCount; i++) {
            const ImageFormat& format = outFormats[i];
            VkFormat vkft = ImageFormat2Vk(format.imageType);
            VulkanTexturePtr texPtr(new VulkanTexture());
            texPtr->InitResource(context, format.width, format.height, vkft,
                                 VK_IMAGE_USAGE_STORAGE_BIT, 0);
            outTexs.push_back(texPtr);
        }
    }
    onInitVkBuffer();
    onInitPipe();
}

bool VkLayer::onFrame() { return true; }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce