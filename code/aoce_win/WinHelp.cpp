#include "WinHelp.hpp"

#include <DirectXMath.h>
#include <comdef.h>
#include <d3d11.h>

#define PI 3.141592653
#define dutorad(X) ((X)/180.0f*PI)
#define radtodu(X) ((X)/PI*180.0f)

using namespace DirectX;

namespace aoce {
namespace win {

bool logHResult(HRESULT hr, const std::string& message, LogLevel level) {
    if (FAILED(hr)) {
        _com_error err(hr);
        LPCTSTR errMsg = err.ErrorMessage();
        std::string msg = message + " hr:" + errMsg;
        logMessage(level, msg);
    }
    return SUCCEEDED(hr);
}

bool validWindow(HWND hwnd) {
    if (hwnd == nullptr || hwnd == INVALID_HANDLE_VALUE) {
        return false;
    }
    RECT rect = {};
    ::GetWindowRect(hwnd, &rect);
    if (rect.bottom <= 0 || rect.right <= 0) {
        return false;
    }
    int32_t height = rect.bottom - rect.top;
    int32_t width = rect.right - rect.left;
    if (height <= 0 || width <= 0) {
        return false;
    }
    return true;
}

XMVECTOR vec2vector(const vec4& float4) {
    return XMVectorSet(float4.x, float4.y, float4.z, float4.w);
}

vec4 vector2vec(const XMVECTOR& mvector) {
    vec4 float4 = {};
    float4.x = XMVectorGetX(mvector);
    float4.y = XMVectorGetY(mvector);
    float4.z = XMVectorGetZ(mvector);
    float4.w = XMVectorGetW(mvector);
    return float4;
}

bool findCameraUE(const vec4* pos, CameraOffset& camera) {
    const int totalIndex = 4;
    XMVECTOR origin[totalIndex];
    for (int i = 0; i < totalIndex; i++) {
        origin[i] = vec2vector(*(pos + i));
    }
    auto direct1 = XMQuaternionNormalize(origin[2] - origin[0]);
    auto direct2 = XMQuaternionNormalize(origin[3] - origin[1]);
    // px=px0+pxd*tx
    // py=py0+pyd*ty
    // px=py,求tx,ty
    auto OfO = origin[3] - origin[2];
    auto DxD = XMVector3Cross(direct1, direct2);
    auto dd = XMVector4Dot(DxD, DxD);
    //得到二条射线距离最近的二点位置。
    auto t1 = XMVector4Dot(XMVector3Cross(OfO, direct2), DxD) / dd;
    auto t2 = XMVector4Dot(XMVector3Cross(OfO, direct1), DxD) / dd;
    // tx,ty对应的世界坐标
    auto p1 = origin[2] + direct1 * t1;
    auto p2 = origin[3] + direct2 * t2;
    // UE4里长度为厘米，如果二点之间长度大于10厘米，那么最好重新定位
    bool bHit = XMVectorGetX(XMVector4Length(p1 - p2)) < 10.0f;
    if (bHit) {
        //摄像机位置
        auto cameraPos = (p1 + p2) / 2.0f;
        //摄像机向前的向量
        auto cameraFor = XMQuaternionNormalize((direct1 + direct2) / 2.0f);
        float totalFov = 0.0f;
        // r = 2*tan(fov/2)*d
        // r为摄像机坐标下点(x,y)与uv(原点在图片中间位置)的值，其r=y/u=(x/u)(cwidth/cheight)
        for (int i = 0; i < totalIndex; i++) {
            auto cpos = origin[i] - cameraPos;
            auto angle = XMVector4AngleBetweenVectors(cpos, cameraFor);
            //得到半角
            auto fov = XMVectorGetX(angle);
            //得到全角
            fov = atan(tan(fov) * 2);
            // UE4的是水平fov,这由垂直转化成水平角度
            fov = atan(tan(fov) * camera.ratio);
            //得全角度
            totalFov += radtodu(fov) * 2.0f;
        }
        //摄像机的fov,fovs测试时，可以输出看下，四个结果应该相关不大，如果很大，肯定算法中出错
        auto fieldOfView = totalFov / 4.0f;
        camera.forward = vector2vec(cameraFor);
        camera.position = vector2vec(cameraPos);
        camera.fov = fieldOfView;
        return true;
    }
    return true;
}

void findPlaneInfoUE(const CameraOffset& camera, const vec4& hmdPos,
                     PlaneInfo& plane) {
    auto hmdVPos = vec2vector(hmdPos);
    auto cmaerVPos = vec2vector(camera.position);
    //得到摄像机的向前向量
    auto cameraForward = vec2vector(camera.forward);
    //得到面片的向前向量,摄像机取反
    auto planForward = -XMQuaternionNormalize(cameraForward);
    //真实人物的面
    auto vplane = XMPlaneFromPointNormal(hmdVPos, planForward);
    //摄像机与真实人物面的交点
    auto planeCenter = XMPlaneIntersectLine(
        vplane, cmaerVPos, cmaerVPos + cameraForward * 1000000);
    //摄像机与交点的距离
    auto distance = planeCenter - cmaerVPos;
    // SIMD数据解压成正常数据
    plane.position = vector2vec(planeCenter);
    plane.forward = vector2vec(planForward);
    //得到面片与摄像机的长度,UE4的FOV是水平方向的
    plane.width = tanf(dutorad(camera.fov / 2.0f)) *
                  XMVectorGetX(XMVector3Length(distance)) * 2.0f;
    plane.height = plane.width / camera.ratio;
}

}  // namespace win
}  // namespace aoce