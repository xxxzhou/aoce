#pragma once

#include "Aoce/Aoce.hpp"
#include "Aoce/Layer/BaseLayer.hpp"

#ifdef _WIN32
#if defined(AOCE_VULKAN_EXTRA_EXPORT_DEFINE)
#define AOCE_VULKAN_EXTRA_EXPORT __declspec(dllexport)
#else
#define AOCE_VULKAN_EXTRA_EXPORT __declspec(dllimport)
#endif
#else
#define AOCE_VULKAN_EXTRA_EXPORT
#endif

namespace aoce {
namespace vulkan {

struct FilterParamet {
    int32_t kernelSizeX = 3;
    int32_t kernelSizeY = 3;
};

AOCE_VULKAN_EXTRA_EXPORT ITLayer<FilterParamet>* createBoxFilterLayer();

}
}
