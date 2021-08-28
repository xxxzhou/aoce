#pragma once

#include "AoceNcnnExport.h"
#include "net.h"

namespace aoce {

AOCE_NCNN_EXPORT ncnn::Mat getMat(uint8_t* data, const ImageFormat& inFormat,
                                  const ImageFormat& outFormat);

}  // namespace aoce