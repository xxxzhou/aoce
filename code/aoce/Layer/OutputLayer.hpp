#pragma once

#include "BaseLayer.hpp"

namespace aoce {

struct OutputParamet {
    int32_t bCpu = true;
    int32_t bGpu = false;
};

struct VkOutGpuTex {
    // vkcommandbuffer
    void* commandbuffer = nullptr;
#if WIN32
    void* image = nullptr;
#elif __ANDROID__
    uint64_t image = 0;
#endif
    int width = 1920;
    int height = 1080;
};

typedef void (*imageProcessAction)(uint8_t* data, ImageFormat imageFormat,
                                   int32_t outIndex);
typedef std::function<void(uint8_t* data, ImageFormat imageFormat,
                           int32_t outIndex)>
    imageProcessHandle;

class ACOE_EXPORT OutputLayer : public ITLayer<OutputParamet> {
   public:
    virtual ~OutputLayer(){};

   protected:
    imageProcessHandle onImageProcessEvent;

   protected:
    inline void onImageProcessHandle(uint8_t* data, ImageFormat imageFormat,
                                     int32_t outIndex = 0) {
        if (onImageProcessEvent) {
            onImageProcessEvent(data,imageFormat, outIndex);
        }
    }

   public:
    inline void setImageProcessHandle(imageProcessHandle handle) {
        onImageProcessEvent = handle;
    };

    // vk: contex表示vkcommandbuffer,texture表示vktexture
    // dx11: contex表示ID3D11Device,texture表示ID3D11Texture2D
    virtual void outVkGpuTex(const VkOutGpuTex& outTex, int32_t outIndex = 0){};

    virtual void outDx11GpuTex(void* device, void* tex){};

#if __ANDROID__
    virtual void outGLGpuTex(const VkOutGpuTex& outTex, uint32_t texType = 0,
                             int32_t outIndex = 0){};
#endif
};
}  // namespace aoce