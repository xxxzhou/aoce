#include "Dx11Resource.hpp"

namespace aoce {
namespace win {

Dx11Resource::~Dx11Resource() { releaseResource(); }

void Dx11Resource::Reset() {
    if (bBufferInit) {
        releaseResource();
        bBufferInit = false;
    }
}

bool Dx11Resource::initResource(ID3D11Device* deviceDx11) {
    Reset();
    return createResource(deviceDx11);
}

void Dx11CSResource::setCpuWrite(bool bCpuWrite) {
    Reset();
    this->bCpuWrite = bCpuWrite;
}

void Dx11CSResource::setShared(bool bShared) {
    Reset();
    this->bShared = bShared;
    if (this->bShared) {
        this->bNoView = true;
    }
}

void Dx11CSResource::setNoView(bool bNoView) {
    Reset();
    this->bNoView = bNoView;
}

void Dx11CSResource::setOnlyUAV(bool bOnlyUAV) {
    Reset();
    this->bOnlyUAV = bOnlyUAV;
}

void Dx11CSResource::releaseResource() {
    srvView.Release();
    uavView.Release();
}

Dx11Texture::~Dx11Texture() {}

void Dx11Texture::setTextureSize(int32_t width, int32_t height,
                                 DXGI_FORMAT format) {
    Reset();
    this->width = width;
    this->height = height;
    this->format = format;
}

bool Dx11Texture::createResource(ID3D11Device* deviceDx11) {
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.Format = format;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.MiscFlags = 0;
    textureDesc.BindFlags =
        D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (bCpuWrite) {
        textureDesc.Usage =
            D3D11_USAGE_DYNAMIC;  // DYNAMIC的对应cpu可写，STAGING
                                  // CPU可读写，否则cpu没任何权限
        textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    }
    if (bShared) {
        textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
    }
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    HRESULT result = 0;
    if (cpuData) {
        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = cpuData;
        result = deviceDx11->CreateTexture2D(&textureDesc, &initData, &texture);
    } else {
        result = deviceDx11->CreateTexture2D(&textureDesc, nullptr, &texture);
    }
    bBufferInit = SUCCEEDED(result);
    if (bBufferInit && !bNoView) {
        if (!bOnlyUAV) {
            createBufferSRV(deviceDx11, texture, &srvView);
        }
        if (!bCpuWrite) {
            createBufferUAV(deviceDx11, texture, &uavView);
        }
    }
    return bBufferInit;
}

bool Dx11Texture::updateResource(ID3D11DeviceContext* ctxDx11) {
    if (!cpuData || !bBufferInit) {
        return false;
    }
    int dataType = width * height * sizeDxFormatElement(format);
    if ((width % 32) != 0) {
        std::string message =
            "texture width " + std::to_string(width) + " no mod 32.";
        logMessage(LogLevel::warn, message.c_str());
    }
    bool bUpdate = updateDx11Resource(ctxDx11, texture, cpuData, dataType);
    return bUpdate;
}

void Dx11Texture::releaseResource() {
    Dx11CSResource::releaseResource();
    texture.Release();
}

Dx11Buffer::~Dx11Buffer() {}

void Dx11Buffer::setBufferSize(int32_t elementSize, int32_t dataType,
                               bool rawBuffer) {
    Reset();
    this->dataType = dataType;
    this->elementSize = elementSize;
    this->bRawBuffer = rawBuffer;
}

bool Dx11Buffer::createResource(ID3D11Device* deviceDx11) {
    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.ByteWidth = elementSize * dataType;
    desc.StructureByteStride = elementSize;  // elementSize;
    // Structured Buffer
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    if (bRawBuffer) {
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    }
    // UAV SAV
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (bCpuWrite) {
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        desc.Usage = D3D11_USAGE_DYNAMIC;
    }
    //可以在不同的上下文使用
    if (bShared) {
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                         D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    }
    HRESULT result = 0;
    if (cpuData) {
        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = cpuData;
        result = deviceDx11->CreateBuffer(&desc, &initData, &buffer);
    } else
        result = deviceDx11->CreateBuffer(&desc, nullptr, &buffer);
    // D3D11 ERROR : ID3D11Device::CreateBuffer : When creating a buffer with
    // the MiscFlag D3D11_RESOURCE_MISC_BUFFER_STRUCTURED specified, the
    // StructureByteStride must be greater than zero, no greater than 2048, and
    // a multiple of 4.[STATE_CREATION ERROR #2097339:
    bBufferInit = SUCCEEDED(result);
    if (bBufferInit && !bNoView) {
        createBufferSRV(deviceDx11, buffer, &srvView);
        if (!bCpuWrite) {
            createBufferUAV(deviceDx11, buffer, &uavView);
        }
    }
    return bBufferInit;
}

bool Dx11Buffer::updateResource(ID3D11DeviceContext* ctxDx11) {
    if (!cpuData || !bBufferInit) return false;
    int elementByteCount = elementSize * dataType;
    bool bUpdate =
        updateDx11Resource(ctxDx11, buffer, cpuData, elementByteCount);
    return bUpdate;
}

void Dx11Buffer::releaseResource() {
    Dx11CSResource::releaseResource();
    buffer.Release();
}

Dx11Constant::~Dx11Constant() {}

void Dx11Constant::setBufferSize(int32_t dataType) { byteDataSize = dataType; }

bool Dx11Constant::createResource(ID3D11Device* deviceDx11) {
    // Constant默认不需要重建功能
    if (buffer != nullptr) return true;
    HRESULT hr;
    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.ByteWidth = (UINT)(ceil(byteDataSize / 16.0)) * 16;
    desc.Usage = D3D11_USAGE_DEFAULT;
    // desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = 0;
    desc.StructureByteStride = 0;
    if (cpuData) {
        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = cpuData;
        initData.SysMemPitch = 0;
        initData.SysMemSlicePitch = 0;
        hr = deviceDx11->CreateBuffer(&desc, &initData, &buffer);
    } else {
        hr = deviceDx11->CreateBuffer(&desc, nullptr, &buffer);
    }
    return SUCCEEDED(hr);
}

bool Dx11Constant::updateResource(ID3D11DeviceContext* ctxDx11) {
    if (!cpuData) return false;
    ctxDx11->UpdateSubresource(buffer, 0, nullptr, cpuData, 0, 0);
    return true;
}

void Dx11Constant::releaseResource() { buffer.Release(); }

Dx11SharedTex::~Dx11SharedTex() { release(); }

bool Dx11SharedTex::restart(ID3D11Device* deviceDx11, int32_t width,
                            int32_t height, DXGI_FORMAT format) {
    texture = std::make_unique<Dx11Texture>();
    texture->setShared(true);
    texture->setTextureSize(width, height, format);
    texture->initResource(deviceDx11);

    sharedHandle = getDx11SharedHandle(texture->texture);
    bGpuUpdate = false;

    return sharedHandle != nullptr;
}

void Dx11SharedTex::release() {}

#pragma region ShaderInclude
ShaderInclude::ShaderInclude(std::string modelName, std::string rctype,
                             int32_t rcId) {
    this->rcId = rcId;
    // readResouce(modelName.c_str(), rcId, rctype.c_str(), strRes, length);
}

HRESULT __stdcall ShaderInclude::Open(D3D_INCLUDE_TYPE IncludeType,
                                      LPCSTR pFileName, LPCVOID pParentData,
                                      LPCVOID* ppData, UINT* pBytes) {
    *ppData = (const void*)strRes.c_str();
    *pBytes = length;
    return true;
}

HRESULT __stdcall ShaderInclude::Close(LPCVOID pData) { return true; }

#pragma endregion
}  // namespace win
}  // namespace aoce