#include "Dx11Helper.hpp"

#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <math.h>

namespace aoce {
namespace win {

DXGI_FORMAT getImageDXFormt(ImageType imageType) {
    DXGI_FORMAT dxFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    switch (imageType) {
        case ImageType::r8:
            dxFormat = DXGI_FORMAT_R8_UNORM;
            break;
        case ImageType::rgba8:
            dxFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
            break;
        case ImageType::r16:
            dxFormat = DXGI_FORMAT_R16_UINT;
            break;
        case ImageType::bgra8:
            dxFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
            break;
        case ImageType::rgba32f:
            dxFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
            break;
        case ImageType::r32f:
            dxFormat = DXGI_FORMAT_R32_FLOAT;
            break;
        case ImageType::r32:
            dxFormat = DXGI_FORMAT_R32_SINT;
            break;
        case ImageType::rgba32:
            dxFormat = DXGI_FORMAT_R32G32B32A32_SINT;
            break;
        default:
            dxFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
            break;
    }
    return dxFormat;
}

bool createDevice11(ID3D11Device** deviceDx11, ID3D11DeviceContext** ctxDx11) {
    *deviceDx11 = nullptr;
    *ctxDx11 = nullptr;

    HRESULT hr = S_OK;
    UINT uCreationFlags = D3D11_CREATE_DEVICE_SINGLETHREADED;
#if defined(DEBUG) || defined(_DEBUG)
    uCreationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flOut;
    static const D3D_FEATURE_LEVEL flvl[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1};  //, D3D_FEATURE_LEVEL_10_1,
                                  // D3D_FEATURE_LEVEL_10_0

    bool bHaveCompute = false;
    hr = D3D11CreateDevice(
        nullptr,                   // Use default graphics card
        D3D_DRIVER_TYPE_HARDWARE,  // Try to create a hardware accelerated
                                   // device
        nullptr,         // Do not use external software rasterizer module
        uCreationFlags,  // Device creation flags
        flvl, sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
        D3D11_SDK_VERSION,  // SDK version
        deviceDx11,         // Device out
        &flOut,             // Actual feature level created
        ctxDx11);           // Context out

    if (SUCCEEDED(hr)) {
        bHaveCompute = true;
        if (flOut < D3D_FEATURE_LEVEL_11_0) {
            // Otherwise, we need further check whether this device support
            // CS4.x (Compute on 10)
            D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
            (*deviceDx11)
                ->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
                                      &hwopts, sizeof(hwopts));
            bHaveCompute =
                hwopts
                    .ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x;
            if (!bHaveCompute) {
                logMessage(
                    LogLevel::error,
                    "No hardware Compute Shader 5.0 capable device found "
                    "(required for doubles), trying to create ref device.");
            }
        }
    } else {
        logHResult(hr, "createDevice11 fail.");
    }
    return SUCCEEDED(hr);
}

bool updateDx11Resource(ID3D11DeviceContext* ctxDx11, ID3D11Resource* resouce,
                        uint8_t* data, uint32_t size) {
    D3D11_MAPPED_SUBRESOURCE mappedResource = {0};
    HRESULT result =
        ctxDx11->Map(resouce, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    // RowPitch 需要注意,如果数据要返回给CPU,经试验,必需是32的倍数
    memcpy(mappedResource.pData, data, size);
    ctxDx11->Unmap(resouce, 0);
    if (!SUCCEEDED(result)) {
        logHResult(result, "updateDx11Resource fail.");
    }
    return SUCCEEDED(result);
}

bool downloadDx11Resource(ID3D11DeviceContext* ctxDx11, ID3D11Resource* resouce,
                          uint8_t** data, uint32_t& byteWidth) {
    D3D11_MAPPED_SUBRESOURCE mappedResource = {};
    HRESULT result =
        ctxDx11->Map(resouce, 0, D3D11_MAP_READ, 0, &mappedResource);
    if (SUCCEEDED(result)) {
        *data = (uint8_t*)mappedResource.pData;
        byteWidth = mappedResource.RowPitch;
    } else {
        logHResult(result, "downloadDx11Resource fail.");
    }
    ctxDx11->Unmap(resouce, 0);
    return SUCCEEDED(result);
}

bool createGUITextureBuffer(ID3D11Device* deviceDx11, int width, int height,
                            ID3D11Texture2D** ppBufOut) {
    *ppBufOut = nullptr;
    D3D11_TEXTURE2D_DESC textureDesc = {0};
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.BindFlags =
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.MiscFlags = D3D11_RESOURCE_MISC_GDI_COMPATIBLE;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    HRESULT result =
        deviceDx11->CreateTexture2D(&textureDesc, nullptr, ppBufOut);
    if (!SUCCEEDED(result)) {
        logHResult(result, "createGUITextureBuffer fail.");
    }
    return SUCCEEDED(result);
}

void copyBufferToRead(ID3D11Device* deviceDx11, ID3D11Buffer* pBuffer,
                      ID3D11Buffer** descBuffer) {
    D3D11_BUFFER_DESC desc = {};
    pBuffer->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    HRESULT result = deviceDx11->CreateBuffer(&desc, nullptr, descBuffer);
    if (SUCCEEDED(result)) {
        CComPtr<ID3D11DeviceContext> ctxDx11 = nullptr;
        deviceDx11->GetImmediateContext(&ctxDx11);
        ctxDx11->CopyResource(*descBuffer, pBuffer);
    } else {
        logHResult(result, "copyBufferToRead buffer fail.");
    }
}

void copyBufferToRead(ID3D11Device* deviceDx11, ID3D11Texture2D* pBuffer,
                      ID3D11Texture2D** descBuffer) {
    D3D11_TEXTURE2D_DESC desc = {};
    pBuffer->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    HRESULT result = deviceDx11->CreateTexture2D(&desc, nullptr, descBuffer);
    if (SUCCEEDED(result)) {
        CComPtr<ID3D11DeviceContext> ctxDx11 = nullptr;
        deviceDx11->GetImmediateContext(&ctxDx11);
        ctxDx11->CopyResource(*descBuffer, pBuffer);
    } else {
        logHResult(result, "copyBufferToRead texture fail.");
    }
}

bool createBufferSRV(ID3D11Device* deviceDx11, ID3D11Buffer* pBuffer,
                     ID3D11ShaderResourceView** ppSRVOut) {
    D3D11_BUFFER_DESC descBuf;
    ZeroMemory(&descBuf, sizeof(descBuf));
    pBuffer->GetDesc(&descBuf);

    D3D11_SHADER_RESOURCE_VIEW_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    desc.BufferEx.FirstElement = 0;

    if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) {
        // This is a Raw Buffer
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        desc.BufferEx.NumElements = descBuf.ByteWidth / 4;
    } else if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) {
        // This is a Structured Buffer
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.BufferEx.NumElements =
            descBuf.ByteWidth / descBuf.StructureByteStride;
    } else {
        return false;
    }

    HRESULT result =
        deviceDx11->CreateShaderResourceView(pBuffer, &desc, ppSRVOut);
    if (!SUCCEEDED(result)) {
        logHResult(result, "buffer createBufferSRV fail.");
    }
    return SUCCEEDED(result);
}

bool createBufferUAV(ID3D11Device* deviceDx11, ID3D11Buffer* pBuffer,
                     ID3D11UnorderedAccessView** ppUAVOut) {
    D3D11_BUFFER_DESC descBuf;
    ZeroMemory(&descBuf, sizeof(descBuf));
    pBuffer->GetDesc(&descBuf);

    D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;

    if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) {
        // This is a Raw Buffer
        desc.Format =
            DXGI_FORMAT_R32_TYPELESS;  // Format must be
                                       // DXGI_FORMAT_R32_TYPELESS, when
                                       // creating Raw Unordered Access View
        desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        desc.Buffer.NumElements = descBuf.ByteWidth / 4;
    } else if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) {
        // This is a Structured Buffer
        desc.Format =
            DXGI_FORMAT_UNKNOWN;  // Format must be must be DXGI_FORMAT_UNKNOWN,
                                  // when creating a View of a Structured Buffer
        desc.Buffer.NumElements =
            descBuf.ByteWidth / descBuf.StructureByteStride;
    } else {
        return false;
    }
    HRESULT result =
        deviceDx11->CreateUnorderedAccessView(pBuffer, &desc, ppUAVOut);
    if (!SUCCEEDED(result)) {
        logHResult(result, "buffer createBufferUAV fail.");
    }
    return SUCCEEDED(result);
}

bool createBufferSRV(ID3D11Device* deviceDx11, ID3D11Texture2D* pBuffer,
                     ID3D11ShaderResourceView** ppSRVOut) {
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    pBuffer->GetDesc(&desc);

    D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
    ZeroMemory(&SRVDesc, sizeof(SRVDesc));
    SRVDesc.Format = desc.Format;
    SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    SRVDesc.Texture2D.MipLevels = 1;

    auto result =
        deviceDx11->CreateShaderResourceView(pBuffer, &SRVDesc, ppSRVOut);
    if (!SUCCEEDED(result)) {
        logHResult(result, "texture createBufferSRV fail.");
    }
    return SUCCEEDED(result);
}

bool createBufferUAV(ID3D11Device* deviceDx11, ID3D11Texture2D* pBuffer,
                     ID3D11UnorderedAccessView** ppUAVOut) {
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    pBuffer->GetDesc(&desc);

    D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
    ZeroMemory(&UAVDesc, sizeof(UAVDesc));
    UAVDesc.Format = desc.Format;
    UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    UAVDesc.Texture2D.MipSlice = 0;

    auto result =
        deviceDx11->CreateUnorderedAccessView(pBuffer, &UAVDesc, ppUAVOut);
    if (!SUCCEEDED(result)) {
        logHResult(result, "texture createBufferUAV fail.");
    }
    return SUCCEEDED(result);
}

HANDLE getDx11SharedHandle(ID3D11Resource* source, bool bNT) {
    HANDLE handle = nullptr;
    // QI IDXGIResource interface to synchronized shared surface.
    CComPtr<IDXGIResource> dxGIResource = nullptr;
    if (!bNT) {
        HRESULT hr = source->QueryInterface(
            __uuidof(IDXGIResource), reinterpret_cast<void**>(&dxGIResource));
        if (FAILED(hr)) {
            logHResult(hr, "get shared texture error.");
            return nullptr;
        }
        // Obtain handle to IDXGIResource object.
        dxGIResource->GetSharedHandle(&handle);
    } else {
        CComPtr<IDXGIResource1> dxGIResource1 = nullptr;
        HRESULT hr = source->QueryInterface(
            __uuidof(IDXGIResource), reinterpret_cast<void**>(&dxGIResource1));
        // DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        dxGIResource1->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr,
                                          &handle);
    }
    return handle;
}

void copySharedToTexture(ID3D11Device* d3ddevice, const HANDLE& sharedHandle,
                         ID3D11Texture2D* texture, bool bNT) {
    if (!d3ddevice) {
        return;
    }
    CComPtr<ID3D11DeviceContext> d3dcontext = nullptr;
    d3ddevice->GetImmediateContext(&d3dcontext);
    if (!d3dcontext) {
        return;
    }
    if (sharedHandle && texture) {
        CComPtr<ID3D11Texture2D> pBuffer = nullptr;
        HRESULT hr = 0;
        if (bNT) {
            CComPtr<ID3D11Device1> d3d11_Device1 = nullptr;
            d3ddevice->QueryInterface(__uuidof(ID3D11Device1),
                                      (void**)&d3d11_Device1);
            hr = d3d11_Device1->OpenSharedResource1(
                sharedHandle, __uuidof(ID3D11Texture2D), (void**)(&pBuffer));
        } else {
            hr = d3ddevice->OpenSharedResource(
                sharedHandle, __uuidof(ID3D11Texture2D), (void**)(&pBuffer));
        }
        if (FAILED(hr) || pBuffer == nullptr) {
            logHResult(hr, "open shared texture error.");
            return;
        }
        CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
        auto hResult = pBuffer->QueryInterface(__uuidof(IDXGIKeyedMutex),
                                               (LPVOID*)&pDX11Mutex);
        if (FAILED(hResult) || pDX11Mutex == nullptr) {
            logHResult(hResult, "get IDXGIKeyedMutex failed.");
            return;
        }
        DWORD result = pDX11Mutex->AcquireSync(AOCE_DX11_MUTEX_READ, 0);
        if (result == WAIT_OBJECT_0 && pBuffer) {
            d3dcontext->CopyResource(texture, pBuffer);
        }
        result = pDX11Mutex->ReleaseSync(AOCE_DX11_MUTEX_WRITE);
    }
}

void copyTextureToShared(ID3D11Device* d3ddevice, const HANDLE& sharedHandle,
                         ID3D11Texture2D* texture, bool bNT) {
    if (!d3ddevice) {
        return;
    }
    CComPtr<ID3D11DeviceContext> d3dcontext = nullptr;
    // ID3D11DeviceContext* d3dcontext = nullptr;
    d3ddevice->GetImmediateContext(&d3dcontext);
    if (!d3dcontext) {
        return;
    }
    if (sharedHandle && texture) {
        CComPtr<ID3D11Texture2D> pBuffer = nullptr;
        HRESULT hr = 0;
        if (bNT) {
            CComPtr<ID3D11Device1> d3d11_Device1 = nullptr;
            d3ddevice->QueryInterface(__uuidof(ID3D11Device1),
                                      (void**)&d3d11_Device1);
            hr = d3d11_Device1->OpenSharedResource1(
                sharedHandle, __uuidof(ID3D11Texture2D), (void**)(&pBuffer));
        } else {
            hr = d3ddevice->OpenSharedResource(
                sharedHandle, __uuidof(ID3D11Texture2D), (void**)(&pBuffer));
        }
        if (FAILED(hr) || pBuffer == nullptr) {
            logHResult(hr, "open shared texture error.");
            return;
        }
        CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
        auto hResult = pBuffer->QueryInterface(__uuidof(IDXGIKeyedMutex),
                                               (LPVOID*)&pDX11Mutex);
        if (FAILED(hResult) || pDX11Mutex == nullptr) {
            logHResult(hResult, "get IDXGIKeyedMutex failed.");
            return;
        }
        DWORD result = pDX11Mutex->AcquireSync(AOCE_DX11_MUTEX_WRITE, 0);
        if (result == WAIT_OBJECT_0 && pBuffer) {
            d3dcontext->CopyResource(pBuffer, texture);
        }
        result = pDX11Mutex->ReleaseSync(AOCE_DX11_MUTEX_READ);
    }
}

int32_t sizeDxFormatElement(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_UINT:
        case DXGI_FORMAT_R32G32B32A32_SINT:
            return 16;

        case DXGI_FORMAT_R32G32B32_TYPELESS:
        case DXGI_FORMAT_R32G32B32_FLOAT:
        case DXGI_FORMAT_R32G32B32_UINT:
        case DXGI_FORMAT_R32G32B32_SINT:
            return 12;

        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_UINT:
        case DXGI_FORMAT_R16G16B16A16_SNORM:
        case DXGI_FORMAT_R16G16B16A16_SINT:
        case DXGI_FORMAT_R32G32_TYPELESS:
        case DXGI_FORMAT_R32G32_FLOAT:
        case DXGI_FORMAT_R32G32_UINT:
        case DXGI_FORMAT_R32G32_SINT:
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
            return 8;

        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UINT:
        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UINT:
        case DXGI_FORMAT_R8G8B8A8_SNORM:
        case DXGI_FORMAT_R8G8B8A8_SINT:
        case DXGI_FORMAT_R16G16_TYPELESS:
        case DXGI_FORMAT_R16G16_FLOAT:
        case DXGI_FORMAT_R16G16_UNORM:
        case DXGI_FORMAT_R16G16_UINT:
        case DXGI_FORMAT_R16G16_SNORM:
        case DXGI_FORMAT_R16G16_SINT:
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_SINT:
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
            return 4;

        case DXGI_FORMAT_R8G8_TYPELESS:
        case DXGI_FORMAT_R8G8_UNORM:
        case DXGI_FORMAT_R8G8_UINT:
        case DXGI_FORMAT_R8G8_SNORM:
        case DXGI_FORMAT_R8G8_SINT:
        case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_D16_UNORM:
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_R16_UINT:
        case DXGI_FORMAT_R16_SNORM:
        case DXGI_FORMAT_R16_SINT:
        case DXGI_FORMAT_B5G6R5_UNORM:
        case DXGI_FORMAT_B5G5R5A1_UNORM:
            return 2;

        case DXGI_FORMAT_R8_TYPELESS:
        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_R8_UINT:
        case DXGI_FORMAT_R8_SNORM:
        case DXGI_FORMAT_R8_SINT:
        case DXGI_FORMAT_A8_UNORM:
            return 1;

            // Compressed format;
            // http://msdn2.microsoft.com/en-us/library/bb694531(VS.85).aspx
        case DXGI_FORMAT_BC2_TYPELESS:
        case DXGI_FORMAT_BC2_UNORM:
        case DXGI_FORMAT_BC2_UNORM_SRGB:
        case DXGI_FORMAT_BC3_TYPELESS:
        case DXGI_FORMAT_BC3_UNORM:
        case DXGI_FORMAT_BC3_UNORM_SRGB:
        case DXGI_FORMAT_BC5_TYPELESS:
        case DXGI_FORMAT_BC5_UNORM:
        case DXGI_FORMAT_BC5_SNORM:
            return 16;

            // Compressed format;
            // http://msdn2.microsoft.com/en-us/library/bb694531(VS.85).aspx
        case DXGI_FORMAT_R1_UNORM:
        case DXGI_FORMAT_BC1_TYPELESS:
        case DXGI_FORMAT_BC1_UNORM:
        case DXGI_FORMAT_BC1_UNORM_SRGB:
        case DXGI_FORMAT_BC4_TYPELESS:
        case DXGI_FORMAT_BC4_UNORM:
        case DXGI_FORMAT_BC4_SNORM:
            return 8;

            // Compressed format;
            // http://msdn2.microsoft.com/en-us/library/bb694531(VS.85).aspx
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
            return 4;

            // These are compressed, but bit-size information is unclear.
        case DXGI_FORMAT_R8G8_B8G8_UNORM:
        case DXGI_FORMAT_G8R8_G8B8_UNORM:
            return 4;

        case DXGI_FORMAT_UNKNOWN:
        default:
            throw new std::exception(
                "Cannot determine format element size; invalid format "
                "specified.");
    }
}

bool createAndCopyToDebugBuf(ID3D11Device* deviceDx11, ID3D11Texture2D* texture,
                             ID3D11Texture2D** debugTex) {
    D3D11_TEXTURE2D_DESC desc = {0};
    texture->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    HRESULT hr = deviceDx11->CreateTexture2D(&desc, nullptr, debugTex);
    if (FAILED(hr)) {
        return false;
    }
    CComPtr<ID3D11DeviceContext> d3dcontext = nullptr;
    deviceDx11->GetImmediateContext(&d3dcontext);
    if (!d3dcontext) {
        return false;
    }
    d3dcontext->CopyResource(texture, *debugTex);
    return true;
}

bool getTextureData(ID3D11Device* deviceDx11, ID3D11Texture2D* texture,
                    uint8_t** data) {
    CComPtr<ID3D11Texture2D> temp = nullptr;
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    texture->GetDesc(&desc);
    if (!(desc.CPUAccessFlags & D3D11_CPU_ACCESS_READ)) {
        bool bCreate = createAndCopyToDebugBuf(deviceDx11, texture, &temp);
        if (!bCreate) {
            return false;
        }
    }
    CComPtr<ID3D11DeviceContext> d3dcontext = nullptr;
    deviceDx11->GetImmediateContext(&d3dcontext);
    if (!d3dcontext) {
        return false;
    }
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    d3dcontext->Map(temp ? temp : texture, 0, D3D11_MAP_READ, 0,
                    &MappedResource);
    *data = (uint8_t*)MappedResource.pData;
    d3dcontext->Unmap(temp ? temp : texture, 0);
    return true;
}

bool getSuitAdapter(IDXGIAdapter1** adapter) {
    *adapter = nullptr;
    CComPtr<IDXGIFactory1> factory = nullptr;
    IID factoryIID = __uuidof(IDXGIFactory1);
    HRESULT hr;
    hr = CreateDXGIFactory1(factoryIID, (void**)&factory);
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture CreateDXGIFactory1 failed");
        return false;
    }
    int32_t index = 0;
    // 枚举适配器
    IDXGIAdapter* pAdapter = nullptr;
    while (factory->EnumAdapters(index, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC adapterDesc = {};
        hr = pAdapter->GetDesc(&adapterDesc);

        ++index;
    }
    return true;
}

}  // namespace win
}  // namespace aoce
