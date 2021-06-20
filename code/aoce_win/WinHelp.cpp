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
    RECT rect = {};
    ::GetWindowRect(hwnd, &rect);
    if (rect.bottom <= 0 || rect.left <= 0 || rect.right <= 0 ||
        rect.top <= 0) {
        return false;
    }
    return true;
}

}  // namespace win
}  // namespace aoce