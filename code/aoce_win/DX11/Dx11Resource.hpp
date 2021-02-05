#pragma once
#include <memory>

#include "Dx11Helper.hpp"
namespace aoce {
namespace win {
class ACOE_WIN_EXPORT Dx11Resource {
   public:
    Dx11Resource(){};
    virtual ~Dx11Resource();

   protected:
    bool bBufferInit = false;

   public:
    uint8_t* cpuData = nullptr;

   public:
    void Reset();
    bool initResource(ID3D11Device* deviceDx11);

   protected:
    virtual bool createResource(ID3D11Device* deviceDx11) = 0;

   public:
    virtual bool updateResource(ID3D11DeviceContext* ctxDx11) { return false; };
    virtual void releaseResource(){};
};

//默认创建一个GPU可读写资源，UAV资源,非共享的
class ACOE_WIN_EXPORT Dx11CSResource : public Dx11Resource {
   public:
    Dx11CSResource(){};
    virtual ~Dx11CSResource(){};

   public:
    CComPtr<ID3D11ShaderResourceView> srvView = nullptr;
    CComPtr<ID3D11UnorderedAccessView> uavView = nullptr;

   protected:
    // cpu的数据可以提交到资源中,如果设定为true,自动不创建对应UAV资源
    bool bCpuWrite = false;
    //是否是可以不同上下文都能访问的资源
    bool bShared = false;
    //是否只创建一个纹理,默认创建相应的SRV/UAV资源
    bool bNoView = false;
    //只创建UAV资源
    bool bOnlyUAV = false;

   public:
    //默认为UAV资源，为ture对应SRV资源
    void setCpuWrite(bool bCpuWrite);
    //默认不共享,为true是共享
    void setShared(bool bShared);
    //默认创建相应UVA/SRV资源，为true不创建
    void setNoView(bool bNoView);
    //
    void setOnlyUAV(bool bOnlyUAV);

   public:
    virtual void releaseResource() override;
};

class ACOE_WIN_EXPORT Dx11Texture : public Dx11CSResource {
   public:
    Dx11Texture(){};
    virtual ~Dx11Texture();

   public:
    // CComQIPtr
    CComPtr<ID3D11Texture2D> texture = nullptr;

   private:
    int32_t width = 0;
    int32_t height = 0;
    // 是否使用NT句柄方式,CUDA不需要,vulkan需要
    bool bNTHandle = false;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;

   public:
    void setTextureSize(int32_t width, int32_t height,
                        DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
                        bool bNT = false);

   protected:
    // 通过 Dx11Resource 继承
    virtual bool createResource(ID3D11Device* deviceDx11) override;

   public:
    // 通过 Dx11Resource 继承
    virtual bool updateResource(ID3D11DeviceContext* ctxDx11) override;
    virtual void releaseResource();
};

class ACOE_WIN_EXPORT Dx11Buffer : public Dx11CSResource {
   public:
    Dx11Buffer(){};
    virtual ~Dx11Buffer();

   public:
    CComPtr<ID3D11Buffer> buffer = nullptr;

   private:
    int32_t elementSize = 0;
    int32_t dataType = 0;
    // Structured Buffer 需要最少四字节
    bool bRawBuffer = false;

   public:
    void setBufferSize(int32_t elementSize, int32_t dataType,
                       bool rawBuffer = false);

   protected:
    // 通过 Dx11Resource 继承
    virtual bool createResource(ID3D11Device* deviceDx11) override;

   public:
    // 通过 Dx11Resource 继承
    virtual bool updateResource(ID3D11DeviceContext* ctxDx11) override;
    virtual void releaseResource();
};

class ACOE_WIN_EXPORT Dx11Constant : public Dx11Resource {
   public:
    Dx11Constant(){};
    virtual ~Dx11Constant();

   public:
    CComPtr<ID3D11Buffer> buffer = nullptr;

   private:
    int32_t byteDataSize = 0;

   public:
    void setBufferSize(int32_t dataType);

   protected:
    // 通过 Dx11Resource 继承
    virtual bool createResource(ID3D11Device* deviceDx11) override;

   public:
    virtual bool updateResource(ID3D11DeviceContext* ctxDx11) override;
    virtual void releaseResource();
};

class ACOE_WIN_EXPORT Dx11SharedTex {
   public:
    Dx11SharedTex(){};
    virtual ~Dx11SharedTex();

   public:
    std::unique_ptr<Dx11Texture> texture = nullptr;
    HANDLE sharedHandle = nullptr;
    bool bGpuUpdate = false;
    bool bNTHandle = false;
   public:
    bool restart(ID3D11Device* deviceDx11, int32_t width, int32_t height,
                 DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,bool bNT= false);
    void release();
};

class ACOE_WIN_EXPORT ShaderInclude : public ID3DInclude {
   public:
    ShaderInclude(std::string modelName, std::string rctype, int32_t rcId);
    ~ShaderInclude(){};

   public:
    // 通过 ID3DInclude 继承
    virtual HRESULT STDMETHODCALLTYPE Open(D3D_INCLUDE_TYPE IncludeType,
                                           LPCSTR pFileName,
                                           LPCVOID pParentData, LPCVOID* ppData,
                                           UINT* pBytes) override;
    virtual HRESULT STDMETHODCALLTYPE Close(LPCVOID pData) override;

   private:
    int32_t rcId = 0;
    std::string strRes = "";
    uint32_t length = 0;
};
}  // namespace win
}  // namespace aoce
