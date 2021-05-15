#pragma once

#include <atlcomcli.h>
#include <d3d11.h>
#include <d3dcommon.h>
#include <d3dcompiler.h>

#include "../WinHelp.hpp"

#define AOCE_DX11_MUTEX_READ 0
#define AOCE_DX11_MUTEX_WRITE 1

namespace aoce {
namespace win {

ACOE_WIN_EXPORT DXGI_FORMAT getImageDXFormt(ImageType imageType);

// 创建一个Dx11环境
ACOE_WIN_EXPORT bool createDevice11(ID3D11Device** deviceDx11,
                                    ID3D11DeviceContext** ctxDx11);
// 从CPU数据传入GPU中
ACOE_WIN_EXPORT bool updateDx11Resource(ID3D11DeviceContext* ctxDx11,
                                        ID3D11Resource* resouce, uint8_t* data,
                                        uint32_t size);
// byteWidth不一定和width*elementSize相等,需要注意处理
ACOE_WIN_EXPORT bool downloadDx11Resource(ID3D11DeviceContext* ctxDx11,
                                          ID3D11Resource* resouce,
                                          uint8_t** data, uint32_t& byteWidth);
// DX11截屏
ACOE_WIN_EXPORT bool createGUITextureBuffer(ID3D11Device* deviceDx11, int width,
                                            int height,
                                            ID3D11Texture2D** ppBufOut);
// 用于创建一块CPU不能读的BUFFER到相同规格CPU能读的BUFFER
ACOE_WIN_EXPORT void copyBufferToRead(ID3D11Device* deviceDx11,
                                      ID3D11Buffer* pBuffer,
                                      ID3D11Buffer** descBuffer);
ACOE_WIN_EXPORT void copyBufferToRead(ID3D11Device* deviceDx11,
                                      ID3D11Texture2D* pBuffer,
                                      ID3D11Texture2D** descBuffer);
// SRV:描述了诸如贴图时,一般用来做CS中的输入,但是ID3D11Texture2D相应的SRV满足特定几种格式也可以做输入,ID3D11Buffer的SRV没限制
// UAV:描述了可以由Shader进行随机读写的显存存空间,用于做CS中的输入与输出
ACOE_WIN_EXPORT bool createBufferSRV(ID3D11Device* deviceDx11,
                                     ID3D11Buffer* pBuffer,
                                     ID3D11ShaderResourceView** ppSRVOut);
ACOE_WIN_EXPORT bool createBufferUAV(ID3D11Device* deviceDx11,
                                     ID3D11Buffer* pBuffer,
                                     ID3D11UnorderedAccessView** ppUAVOut);
ACOE_WIN_EXPORT bool createBufferSRV(ID3D11Device* deviceDx11,
                                     ID3D11Texture2D* pBuffer,
                                     ID3D11ShaderResourceView** ppSRVOut);
ACOE_WIN_EXPORT bool createBufferUAV(ID3D11Device* deviceDx11,
                                     ID3D11Texture2D* pBuffer,
                                     ID3D11UnorderedAccessView** ppUAVOut);

ACOE_WIN_EXPORT HANDLE getDx11SharedHandle(ID3D11Resource* source,
                                           bool bNT = false);

ACOE_WIN_EXPORT void copySharedToTexture(ID3D11Device* d3ddevice,
                                         const HANDLE& sharedHandle,
                                         ID3D11Texture2D* texture,
                                         bool bNT = false);

ACOE_WIN_EXPORT void copyTextureToShared(ID3D11Device* d3ddevice,
                                         const HANDLE& sharedHandle,
                                         ID3D11Texture2D* texture,
                                         bool bNT = false);

ACOE_WIN_EXPORT int32_t sizeDxFormatElement(DXGI_FORMAT format);
}  // namespace win
}  // namespace aoce