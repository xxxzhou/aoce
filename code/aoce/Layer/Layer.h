#pragma once

#include "LayerGraph.hpp"

extern "C" {
ACOE_EXPORT aoce::LayerGraph* CreateProcess(AOCE_GPU_SDK gpuType);
// 外部扩展层导出给C提供一个crate,一个update参数
ACOE_EXPORT aoce::InputLayer* CreateInputLayer(AOCE_GPU_SDK gpuType);
}