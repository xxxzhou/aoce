#include "CudaHelper.hpp"

namespace aoce {
namespace cuda {
int32_t ImageFormat2Cuda(ImageType imageType) {
    switch (imageType) {
        case ImageType::bgra8:
            return AOCE_CV_8UC4;
        case ImageType::r16:
            return AOCE_CV_16UC1;
        case ImageType::r8:
            return AOCE_CV_8UC1;  // VK_FORMAT_R8_UINT VK_FORMAT_S8_UINT
                                  // VK_FORMAT_R8_UNORM
        case ImageType::rgba8:
            return AOCE_CV_8UC4;
        default:
            return AOCE_CV_8UC1;
    }
}
#if WIN32
void setNppStream(cudaStream_t& stream) {
    auto oldStream = nppGetStream();
    if (oldStream != stream) {
        cudaDeviceSynchronize();
        nppSetStream(stream);
    }
};

//把CUDA资源复制给DX11纹理(这里有个算是我遇到最奇怪的BUG之一,有行中文注释会导致这函数不能运行?)
void gpuMat2D3dTexture(CudaMatRef frame, Dx11CudaResource& cudaResource,
                       cudaStream_t stream) {
    if (cudaResource.texture != nullptr) {
        // cuda map dx11,资源数组间map
        cudaError_t cerror =
            cudaGraphicsMapResources(1, &cudaResource.cudaResource, stream);
        // map单个资源 cuda->(dx11 bind cuda resource)
        cerror = cudaGraphicsSubResourceGetMappedArray(
            &cudaResource.cuArray, cudaResource.cudaResource, 0, 0);
        cerror = cudaMemcpy2DToArray(
            cudaResource.cuArray, 0, 0, frame->ptr(), frame->getStep(),
            frame->getWidth() * sizeof(int32_t), frame->getHeight(),
            cudaMemcpyDeviceToDevice);
        // cuda unmap dx11
        cerror =
            cudaGraphicsUnmapResources(1, &cudaResource.cudaResource, stream);
    }
};

//把与DX11资源绑定显存复制出来
void d3dTexture2GpuMat(CudaMatRef frame, Dx11CudaResource& cudaResource,
                       cudaStream_t stream) {
    if (cudaResource.texture != nullptr) {
        // cuda map dx11,资源数组间map
        cudaGraphicsMapResources(1, &cudaResource.cudaResource, stream);
        // map单个资源 (dx11 bind cuda resource)->cuda
        cudaGraphicsSubResourceGetMappedArray(&cudaResource.cuArray,
                                              cudaResource.cudaResource, 0, 0);
        cudaMemcpy2DFromArray(frame->ptr(), frame->getStep(),
                              cudaResource.cuArray, 0, 0,
                              frame->getWidth() * sizeof(int32_t),
                              frame->getHeight(), cudaMemcpyDeviceToDevice);
        // cuda unmap dx11
        cudaGraphicsUnmapResources(1, &cudaResource.cudaResource, stream);
    }
};

//绑定一个DX11共享资源与CUDA资源,分别在DX11与CUDA都有相关引用,二边分别可读可写
bool registerCudaResource(Dx11CudaResource& cudaDx11,
                          std::shared_ptr<win::Dx11SharedTex> sharedResource,
                          ID3D11Device* device, int32_t width, int32_t height,
                          DXGI_FORMAT format) {
    cudaDx11.unBind();
    bool bInit = sharedResource->restart(device, width, height, format);
    if (bInit) {
        cudaDx11.texture = sharedResource->texture->texture;
        cudaError_t result = cudaGraphicsD3D11RegisterResource(
            &cudaDx11.cudaResource, cudaDx11.texture,
            cudaGraphicsRegisterFlagsNone);
        if (result != cudaSuccess) {
            logMessage(LogLevel::error,
                       "cudaGraphicsD3D11RegisterResource fails.");
        }
    }
    return bInit;
}
#endif

void reCudaAllocCpu(void** data, int32_t length) {
    if (*data != nullptr) {
        cudaFreeHost(*data);
        *data = nullptr;
    }
    cudaHostAlloc(data, length, cudaHostAllocDefault);
}

void reCudaAllocGpu(void** data, int32_t length) {
    if (*data != nullptr) {
        cudaFree(*data);
        *data = nullptr;
    }
    cudaMalloc(data, length);
}

}  // namespace cuda
}  // namespace aoce