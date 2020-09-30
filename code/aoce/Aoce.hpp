#pragma once
#include <string>

#include "Aoce.h"
// #include "Module/IModule.hpp"
// #include "VideoDevice/VideoManager.hpp"
// #include "PipeGraph/PipeGraph.hpp"

ACOE_EXPORT void logMessage(aoce::LogLevel level, const std::string& message);

ACOE_EXPORT std::wstring utf8TWstring(const std::string& str);
ACOE_EXPORT std::string utf8TString(const std::wstring& str);

ACOE_EXPORT void copywcharstr(wchar_t* dest, const wchar_t* source, int32_t maxlength);

ACOE_EXPORT void copycharstr(char* dest, const char* source, int32_t maxlength);
