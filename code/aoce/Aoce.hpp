#pragma once
#include <assert.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Aoce.h"
#include "AoceCore.h"

namespace aoce {

// 如果列表的顺序无关,可以尝试如下三种的快速删除,不需要move数据
template <typename T>
void removeItem(std::vector<T>& vect, const T& t) {
    if (vect.empty()) {
        return;
    }
    auto size = vect.size();
    if (vect[size - 1] == t) {
        vect.pop_back();
        return;
    }
    auto iter = std::find(vect.begin(), vect.end(), t);
    if (iter != vect.end()) {
        *iter = vect[size - 1];
        vect.pop_back();
    }
};

// UE4 std::find_if not find
#if WIN32
template <typename T>
void removeItem(std::vector<T>& vect, std::function<bool(T t)> pred) {
    if (vect.empty()) {
        return;
    }
    auto iter = std::find_if(vect.begin(), vect.end(), pred);
    if (iter != vect.end()) {
        *iter = vect[vect.size() - 1];
        vect.pop_back();
    }
};
#endif

template <typename T>
void removeItem(std::vector<T>& vect, typename std::vector<T>::iterator iter) {
    if (vect.empty()) {
        return;
    }
    if (iter != vect.end()) {
        *iter = vect[vect.size() - 1];
        vect.pop_back();
    }
};

// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
// template <typename... Args>
// std::string string_format(const std::string& format, Args... args) {
//     size_t size = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
//     if (size <= 0) {
//         return "";
//     }
//     std::unique_ptr<char[]> buf(new char[size]);
//     std::snprintf(buf.get(), size, format.c_str(), args...);
//     return std::string(buf.get(),
//                        buf.get() + size - 1);  // We don't want the '\0'
// }

template <typename T>
void string_format(std::ostream& o, T t) {
    o << t;
}

template <typename T, typename... Args>
void string_format(std::ostream& o, T t, Args... args) {
    string_format(o, t);
    string_format(o, args...);
}

template <typename... Args>
void string_format(std::string& msg, Args... args) {
    std::ostringstream oss;
    string_format(oss, args...);
    msg = oss.str();
}

// 回调定义,在aoce下的模块可以使用C++传值改用std::function.
// typedef C++ function后缀定义 handle,C为action,前缀不要加on
// 定义的handle变量为on前缀,event后缀
// 定义的handle包装方法为on前缀,handle后缀
// 定义的set handle变量的方法为set前缀(不包含on),handle后缀
typedef std::function<void(int32_t level, const char* message)> logEventHandle;

ACOE_EXPORT void setLogHandle(logEventHandle action);

ACOE_EXPORT void logMessage(aoce::LogLevel level, const std::string& message);

// 如果结果不正确,先输出提示,然后assert
ACOE_EXPORT void logAssert(bool expression, const std::string& message);

ACOE_EXPORT std::wstring utf8TWstring(const std::string& str);
ACOE_EXPORT std::string utf8TString(const std::wstring& str);

ACOE_EXPORT void copywcharstr(wchar_t* dest, const wchar_t* source,
                              int32_t maxlength);

ACOE_EXPORT void copycharstr(char* dest, const char* source, int32_t maxlength);

ACOE_EXPORT std::string getAocePath();

//针对AudioDesc的bitsize为16，并且内部格式为AV_SAMPLE_FMT_S16的保证无问题，P格式肯定不行
ACOE_EXPORT void getWavHeader(std::vector<uint8_t>& header, uint32_t dataSize,
                              const AudioFormat& audioDesc);

#if WIN32
ACOE_EXPORT bool existsFile(const wchar_t* filePath);

ACOE_EXPORT bool loadFileBinary(const wchar_t* filePath,
                                std::vector<uint8_t>& data);

// 默认文件不存在 则建立新文件 文件存在则直接清空文件内容
ACOE_EXPORT bool saveFileBinary(const wchar_t* filePath, void* data,
                                int32_t lenght);
#endif
}  // namespace aoce