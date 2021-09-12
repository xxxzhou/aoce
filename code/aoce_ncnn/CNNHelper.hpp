#pragma once

#include "AoceNcnnExport.h"
#include "aoce_vulkan/VkCommand.hpp"
#include "aoce_vulkan/vulkan/VulkanBuffer.hpp"
#include "net.h"

namespace aoce {

AOCE_NCNN_EXPORT ncnn::Mat getMat(uint8_t* data, const ImageFormat& inFormat,
                                  const ImageFormat& outFormat);

void testVkMat(ncnn::VkMat& mat);

int32_t loadNet(ncnn::Net* net, const std::string& paramFile,
                const std::string& modelFile);

void copyBuffer(ncnn::Net* net, ncnn::VkMat& dstMat,
                aoce::vulkan::VulkanBuffer* srcbuffer);

}  // namespace aoce