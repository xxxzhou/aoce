#pragma once
#include <atlcomcli.h>  // comptr

#include <Aoce.hpp>
#include <string>

#ifdef _MSC_VER
#if AOCE_WIN_EXPORT_DEFINE
#define ACOE_WIN_EXPORT __declspec(dllexport)
#else
#define ACOE_WIN_EXPORT __declspec(dllimport)
#endif
#else
#define ACOE_WIN_EXPORT
#endif

namespace aoce {
namespace win {

struct CameraOffset {
    vec4 position;
    vec4 forward;
    float fov;
    float ratio;
};

struct PlaneInfo {
    vec4 position;
    vec4 forward;
    float width;
    float height;
};

ACOE_WIN_EXPORT bool logHResult(HRESULT hr, const std::string& message,
                                LogLevel level = LogLevel::error);
ACOE_WIN_EXPORT bool validWindow(HWND hwnd);

typedef std::function<void(void* device, void* backTex)> onTickHandle;

//四点对位功能，传入四点位置，以及camera里的长宽比ratio信息，得到摄像机的虚拟位置，fov等信息
bool findCameraUE(const vec4* pos, CameraOffset& camera);
//传入摄像机的虚拟位置与fov及对应的HMD位置，得到所需生成面片的位置与大小信息
void findPlaneInfoUE(const CameraOffset& camera, const vec4& hmd,
                     PlaneInfo& plane);

}  // namespace win
}  // namespace aoce