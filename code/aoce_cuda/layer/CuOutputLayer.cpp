#include "CuOutputLayer.hpp"

#include "../CudaHelper.hpp"
#include "CuPipeGraph.hpp"

using namespace aoce::win;

namespace aoce {
namespace cuda {

CuOutputLayer::CuOutputLayer(/* args */) {
    bOutput = true;
    shardTex = std::make_shared<Dx11SharedTex>();
}

CuOutputLayer::~CuOutputLayer() { cudaResoure.unBind(); }

void CuOutputLayer::onInitLayer() {
    if (paramet.bCpu) {
        int32_t imageSize = getImageTypeSize(inFormats[0].imageType);
        cpuData.resize(inFormats[0].width * inFormats[0].height * imageSize);
    }
    if (paramet.bGpu) {
        DXGI_FORMAT dxFormat = getImageDXFormt(inFormats[0].imageType);
        // 绑定一个DX11共享资源与CUDA资源
        registerCudaResource(cudaResoure, shardTex,
                             cuPipeGraph->getDX11Device(), inFormats[0].width,
                             inFormats[0].height, dxFormat);
    }
}

bool CuOutputLayer::onFrame() {
    if (paramet.bCpu) {
        inTexs[0]->download(cpuData.data(), 0, stream);
        cudaStreamSynchronize(stream);
        onImageProcessHandle(cpuData.data(), inFormats[0], 0);
    }
    // 把最新结果写入与DX11共享资源的CUDA资源
    if (paramet.bGpu) {
        if (shardTex->texture == nullptr) {
            return true;
        }
        cudaStreamSynchronize(stream);
        CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
        HRESULT hResult = shardTex->texture->texture->QueryInterface(
            __uuidof(IDXGIKeyedMutex), (LPVOID*)&pDX11Mutex);
        DWORD result = pDX11Mutex->AcquireSync(AOCE_DX11_MUTEX_WRITE, 0);
        if (result == WAIT_OBJECT_0) {
            gpuMat2D3dTexture(inTexs[0], cudaResoure, stream);
        }
        result = pDX11Mutex->ReleaseSync(AOCE_DX11_MUTEX_READ);
        shardTex->bGpuUpdate = true;
    }
    return true;
}

void CuOutputLayer::onInitCuBufffer() { onFormatChanged(outFormats[0], 0); }

void CuOutputLayer::onUpdateParamet() {
    if (paramet.bCpu == oldParamet.bCpu && paramet.bGpu == oldParamet.bGpu) {
        return;
    }
    resetGraph();
}

void CuOutputLayer::outDx11GpuTex(void* device, void* tex) {
    if (!paramet.bGpu) {
        return;
    }
    ID3D11Device* dxdevice = (ID3D11Device*)device;
    ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)tex;
    if (dxdevice == nullptr || dxtexture == nullptr) {
        return;
    }
    // 把DX11共享资源复制到另一线程上的device的上纹理
    if (shardTex && shardTex->bGpuUpdate) {
        copySharedToTexture(dxdevice, shardTex->sharedHandle, dxtexture);
        shardTex->bGpuUpdate = false;
    }
}

}  // namespace cuda
}  // namespace aoce
