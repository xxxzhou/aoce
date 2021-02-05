#include "CuInputLayer.hpp"

#include "../CudaHelper.hpp"
#include "CuPipeGraph.hpp"

using namespace aoce::win;

namespace aoce {
namespace cuda {

extern void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest,
                         cudaStream_t stream);
extern void argb2rgba_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest,
                          cudaStream_t stream);

CuInputLayer::CuInputLayer(/* args */) { bInput = true; }

CuInputLayer::~CuInputLayer() {}

void CuInputLayer::onUpdateParamet() {
    if (pipeGraph) {
        pipeGraph->reset();
    }
}

void CuInputLayer::onInitCuBufffer() {
    if (videoFormat.videoType == VideoType::rgb8) {
        tempMat = std::shared_ptr<CudaMat>(new CudaMat());
        tempMat->create(inFormats[0].width, inFormats[0].height, AOCE_CV_8UC3);
    } else if (videoFormat.videoType == VideoType::bgra8) {
        tempMat = std::shared_ptr<CudaMat>(new CudaMat());
        tempMat->create(inFormats[0].width, inFormats[0].height, AOCE_CV_8UC4);
    }
    if (paramet.bGpu) {
        if (!device) {
            createDevice11(&device, &ctx);
            shardTex = std::make_shared<Dx11SharedTex>();
        }
        DXGI_FORMAT dxFormat = getImageDXFormt(inFormats[0].imageType);
        // 绑定一个DX11共享资源与CUDA资源
        registerCudaResource(cudaResoure, shardTex, device, inFormats[0].width,
                             inFormats[0].height, dxFormat);
    }
}

bool CuInputLayer::onFrame() {
    if (paramet.bCpu) {
        if (this->videoFormat.videoType == VideoType::rgb8) {
            tempMat->upload(frameData, 0, stream);
            rgb2rgba_gpu(*tempMat, *outTexs[0], stream);

        } else if (videoFormat.videoType == VideoType::bgra8) {
            tempMat->upload(frameData, 0, stream);
            argb2rgba_gpu(*tempMat, *outTexs[0], stream);
        } else {
            outTexs[0]->upload(frameData, 0, stream);
        }
    }
    if (paramet.bGpu) {
        if (shardTex->texture == nullptr) {
            return true;
        }
        CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
        HRESULT hResult = shardTex->texture->texture->QueryInterface(
            __uuidof(IDXGIKeyedMutex), (LPVOID*)&pDX11Mutex);
        DWORD result = pDX11Mutex->AcquireSync(AOCE_DX11_MUTEX_READ, 0);
        if (result == WAIT_OBJECT_0) {
            if (this->videoFormat.videoType == VideoType::rgb8) {
                d3dTexture2GpuMat(tempMat, cudaResoure, stream);
                rgb2rgba_gpu(*tempMat, *outTexs[0], stream);
            } else if (videoFormat.videoType == VideoType::bgra8) {
                d3dTexture2GpuMat(tempMat, cudaResoure, stream);
                argb2rgba_gpu(*tempMat, *outTexs[0], stream);
            } else {
                d3dTexture2GpuMat(outTexs[0], cudaResoure, stream);
            }
            shardTex->bGpuUpdate = false;
        }
        result = pDX11Mutex->ReleaseSync(AOCE_DX11_MUTEX_WRITE);
    }
    return true;
}

void CuInputLayer::onInputGpuDx11(void* device, void* tex) {
    ID3D11Device* dxdevice = (ID3D11Device*)device;
    ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)tex;
    if (!paramet.bGpu || dxdevice == nullptr || dxtexture == nullptr) {
        return;
    }
    copyTextureToShared(dxdevice, shardTex->sharedHandle, dxtexture);
    shardTex->bGpuUpdate = true;
}

}  // namespace cuda
}  // namespace aoce