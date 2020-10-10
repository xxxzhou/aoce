#pragma once
#include <atlcomcli.h>  // comptr

#include <Aoce.hpp>
#include <string>

#ifdef _MSC_VER
#if defined(AOCE_WIN_EXPORT_DEFINE)
#define ACOE_WIN_EXPORT __declspec(dllexport)
#else
#define ACOE_WIN_EXPORT __declspec(dllimport)
#endif
#else
#define ACOE_WIN_EXPORT
#endif
namespace aoce {

namespace win {
ACOE_WIN_EXPORT bool logHResult(HRESULT hr, const std::string& message,
                                LogLevel level = LogLevel::error);
}
}  // namespace aoce