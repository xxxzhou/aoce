#include "WinHelp.hpp"

#include <comdef.h>
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

}  // namespace win
}  // namespace aoce