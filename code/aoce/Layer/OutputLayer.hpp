#pragma once

#include "BaseLayer.hpp"

namespace aoce {

class ACOE_EXPORT OutputLayer : public IOutputLayer {
    friend class PipeGraph;

   public:
    virtual ~OutputLayer(){};

   protected:
    IOutputLayerObserver* observer = nullptr;

   protected:
    void onImageProcessHandle(uint8_t* data, const ImageFormat& imageFormat,
                              int32_t outIndex = 0);
    void onFormatChanged(const ImageFormat& imageFormat, int32_t outIndex = 0);

   public:
    virtual void setObserver(IOutputLayerObserver* observer) final;

    // vk: contex表示vkcommandbuffer,texture表示vktexture
    // dx11: contex表示ID3D11Device,texture表示ID3D11Texture2D
    virtual void outVkGpuTex(const VkOutGpuTex& outTex,
                             int32_t outIndex = 0) override{};

    virtual void outDx11GpuTex(void* device, void* tex) override{};

#if __ANDROID__
    virtual void outGLGpuTex(const VkOutGpuTex& outTex, uint32_t texType = 0,
                             int32_t outIndex = 0) override{};
#endif
};
}  // namespace aoce