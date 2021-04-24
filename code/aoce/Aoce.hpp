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

ACOE_EXPORT void logMessage(aoce::LogLevel level, const std::string& message);

ACOE_EXPORT void logAssert(bool expression, const std::string& message);

ACOE_EXPORT std::wstring utf8TWstring(const std::string& str);
ACOE_EXPORT std::string utf8TString(const std::wstring& str);

ACOE_EXPORT void copywcharstr(wchar_t* dest, const wchar_t* source,
                              int32_t maxlength);

ACOE_EXPORT void copycharstr(char* dest, const char* source, int32_t maxlength);

ACOE_EXPORT aoce::ImageType videoType2ImageType(
    const aoce::VideoType& videoType);

// 原则上,应该只由VideoType转ImageType
// ImageType转VideoType,只有bgra8/r16/rgba8三种有意义
ACOE_EXPORT aoce::VideoType imageType2VideoType(
    const aoce::ImageType& imageType);

ACOE_EXPORT int32_t getYuvIndex(const aoce::VideoType& videoType);

ACOE_EXPORT aoce::ImageFormat videoFormat2ImageFormat(
    const aoce::VideoFormat& videoFormat);

ACOE_EXPORT int32_t getImageTypeSize(const aoce::ImageType& imageType);

// 平面格式可能非紧密排列,给GPU的紧密排列大小,否则返回0
ACOE_EXPORT int32_t getVideoFrame(const aoce::VideoFrame& frame,
                                  uint8_t* data = nullptr);

ACOE_EXPORT std::string getAocePath();

#if WIN32
ACOE_EXPORT bool existsFile(const wchar_t* filePath);

ACOE_EXPORT bool loadFileBinary(const wchar_t* filePath,
                                std::vector<uint8_t>& data);

// 默认文件不存在 则建立新文件 文件存在则直接清空文件内容
ACOE_EXPORT bool saveFileBinary(const wchar_t* filePath, void* data,
                                int32_t lenght);
#endif
