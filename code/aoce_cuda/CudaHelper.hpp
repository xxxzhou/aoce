#pragma once
#include <npp.h>

#include <Aoce.hpp>
#include <memory>

#include "CudaMat.hpp"
#if WIN32
#include <cuda_d3d11_interop.h>
#include <d3d11.h>

#include "../aoce_win/DX11/Dx11Resource.hpp"
#endif

#define SAFE_AOCE_CUDA_DELETE(p) \
    {                            \
        if (p) {                 \
            cudaFree(p);         \
            p = nullptr;         \
        }                        \
    }

namespace aoce {
namespace cuda {

int32_t ImageFormat2Cuda(ImageType imageType);

#if WIN32
struct Dx11CudaResource {
    // cuda资源
    cudaGraphicsResource* cudaResource = nullptr;
    // cuda具体资源
    cudaArray* cuArray = nullptr;
    //对应DX11纹理
    ID3D11Texture2D* texture = nullptr;
    //对应CPU数据
    uint8_t* cpuData = nullptr;
    //取消绑定
    void unBind() {
        if (cudaResource != nullptr && texture != nullptr) {
            cudaGraphicsUnregisterResource(cudaResource);
            cudaResource = nullptr;
        }
    }
};

void AOCE_CUDA_EXPORT setNppStream(cudaStream_t& stream);

// 把CUDA资源复制给DX11纹理(这里有个算是我遇到最奇怪的BUG之一,有行中文注释会导致这函数不能运行?)
void AOCE_CUDA_EXPORT gpuMat2D3dTexture(CudaMatRef frame, Dx11CudaResource& cudaResource,
                       cudaStream_t stream);

// 把与DX11资源绑定显存复制出来
void AOCE_CUDA_EXPORT d3dTexture2GpuMat(CudaMatRef frame, Dx11CudaResource& cudaResource,
                       cudaStream_t stream);

// 绑定一个DX11共享资源与CUDA资源,分别在DX11与CUDA都有相关引用,二边分别可读可写
bool AOCE_CUDA_EXPORT registerCudaResource(Dx11CudaResource& cudaDx11,
                          std::shared_ptr<win::Dx11SharedTex> sharedResource,
                          ID3D11Device* device, int32_t width, int32_t height,
                          DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM);
#endif

void AOCE_CUDA_EXPORT reCudaAllocCpu(void** data, int32_t length);

void AOCE_CUDA_EXPORT reCudaAllocGpu(void** data, int32_t length);

}  // namespace cuda
}  // namespace aoce