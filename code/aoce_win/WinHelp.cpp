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
}  // namespace win
}  // namespace aoce