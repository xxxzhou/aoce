#pragma once

#include "BaseLayer.hpp"

namespace aoce {

struct OutputParamet {
    int32_t bCpu = true;
    int32_t bGpu = false;
};

class OutputLayer : public ILayer<OutputParamet> {};
}  // namespace aoce