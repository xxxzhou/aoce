#include "CuOutputLayer.hpp"

#include "../CudaHelper.hpp"
#include "CuPipeGraph.hpp"

using namespace aoce::win;

namespace aoce {
namespace cuda {

CuOutputLayer::CuOutputLayer(/* args */) { bOutput = true; }

CuOutputLayer::~CuOutputLayer() { cudaResoure.unBind(); }

void CuOutputLayer::onInitLayer() {
    if (paramet.bCpu) {
        int32_t imageSize = getImageTypeSize(inFormats[0].imageType);
        cpuData.resize(inFormats[0].width * inFormats[0].height * imageSize);
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

bool CuOutputLayer::onFrame() {
    if (paramet.bCpu) {
        inTexs[0]->download(cpuData.data(), 0, stream);
        cudaStreamSynchronize(stream);
        onImageProcessHandle(cpuData.data(), inFormats[0].width,
                             inFormats[0].height, 0);
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
        DWORD result = pDX11Mutex->AcquireSync(0, 0);
        if (result == WAIT_OBJECT_0) {
            gpuMat2D3dTexture(inTexs[0], cudaResoure, stream);
        }
        result = pDX11Mutex->ReleaseSync(1);
        shardTex->bGpuUpdate = true;
    }
    return true;
}

void CuOutputLayer::onInitCuBufffer() {}

void CuOutputLayer::onUpdateParamet() { pipeGraph->reset(); }

void CuOutputLayer::outDx11GpuTex(void* device, void* tex) {
    ID3D11Device* dxdevice = (ID3D11Device*)device;
    ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)tex;
    if (!paramet.bGpu || dxdevice == nullptr || dxtexture == nullptr) {
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
