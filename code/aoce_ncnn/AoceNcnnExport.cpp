#include "CNNHelper.hpp"
#include "FaceDetector.hpp"
#include "FaceKeypointDetector.hpp"
#include "VideoMatting.hpp"
#include "VkNcnnInLayer.hpp"
#include "aoce/AoceManager.hpp"

namespace aoce {

int32_t getPixelType(ImageType imageType) {
    switch (imageType) {
        case ImageType::bgra8:
            return ncnn::Mat::PIXEL_BGRA;
        case ImageType::rgba8:
            return ncnn::Mat::PIXEL_RGBA;
        case ImageType::r8:
            return ncnn::Mat::PIXEL_GRAY;
        case ImageType::bgr8:
            return ncnn::Mat::PIXEL_BGR;
        case ImageType::rgb8:
            return ncnn::Mat::PIXEL_RGB;
        default:
            return 0;
    }
}

ncnn::Mat getMat(uint8_t* data, const ImageFormat& inFormat,
                 const ImageFormat& outFormat) {
    ncnn::Mat in = {};
    int32_t inPixel = getPixelType(inFormat.imageType);
    int32_t outPixel = getPixelType(outFormat.imageType);
    if (inPixel > 0) {
        ncnn::Mat::PixelType pixelType = (ncnn::Mat::PixelType)inPixel;
        if (inPixel != outPixel) {
            pixelType = (ncnn::Mat::PixelType)(inPixel | (outPixel << 16));
        }
        if (inFormat.width == outFormat.width &&
            inFormat.height == outFormat.height) {
            in = ncnn::Mat::from_pixels(data, pixelType, inFormat.width,
                                        inFormat.height);
        } else {
            in = ncnn::Mat::from_pixels_resize(data, pixelType, inFormat.width,
                                               inFormat.height, outFormat.width,
                                               outFormat.height);
        }
    }
    return in;
}

void testVkMat(ncnn::VkMat& mat) {
#if AOCE_DEBUG_TYPE
    float* data = (float*)mat.mapped_ptr();
    float xx = 0.0f;
    for (int32_t i = 0; i < mat.c; i++) {
        int32_t size = mat.w * mat.h;
        std::vector<float> fd(size, 0.0f);
        memcpy(fd.data(), data + i * size, size);        
    }
#endif
}

int32_t loadNet(ncnn::Net* net, const std::string& paramFile,
                const std::string& modelFile) {
#if defined(__ANDROID__)
    AAssetManager* assetManager = AoceManager::Get().getAppEnv().assetManager;
    assert(assetManager != nullptr);
    int32_t ret = net->load_param(assetManager, paramFile.c_str());
    if (ret == 0) {
        ret = net->load_model(assetManager, modelFile.c_str());
    }
#else
    std::string paramPath = getAocePath() + "/" + paramFile;
    std::string modelPath = getAocePath() + "/" + modelFile;
    int32_t ret = net->load_param(paramPath.c_str());
    if (ret == 0) {
        ret = net->load_model(modelPath.c_str());
    }
#endif
    return ret;
}

void copyBuffer(ncnn::Net* net, ncnn::VkMat& dstMat,
                aoce::vulkan::VulkanBuffer* buffer) {
    // ncnn::VkBufferMemory nBuffer = {};
    // nBuffer.buffer = buffer->buffer;
    // nBuffer.memory = buffer->memory;
    // nBuffer.offset = 0;
    // nBuffer.capacity = buffer->getBufferSize();
    // nBuffer.mapped_ptr = buffer->getCpuData();
    // nBuffer.access_flags = VK_ACCESS_SHADER_READ_BIT;
    // nBuffer.stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    // nBuffer.refcount = 1;

    // const ncnn::VulkanDevice* device = net->vulkan_device();
    // ncnn::VkAllocator* vkallocator = device->acquire_blob_allocator();
    // ncnn::VkCompute cmd(device);
    // ncnn::VkMat nMat(netFormet.width, netFormet.height, 3, &nBuffer, 4,
    //                  vkallocator);
    // ncnn::Option opt = {};
    // opt.blob_vkallocator = vkallocator;
    // opt.workspace_vkallocator = vkallocator;
    // opt.staging_vkallocator = vkallocator;
    // cmd.record_clone(nMat, dstMat, opt);
    // cmd.submit_and_wait();
}

IFaceDetector* createFaceDetector() {
    FaceDetector* detector = new FaceDetector();
    return detector;
}

IFaceKeypointDetector* createFaceKeypointDetector() {
    FaceKeypointDetector* detector = new FaceKeypointDetector();
    return detector;
}

IBaseLayer* createNcnnInLayer() {
    VkNcnnInLayer* layer = new VkNcnnInLayer();
    return layer;
}

INcnnInCropLayer* createNcnnInCropLayer() {
    VkNcnnInCropLayer* layer = new VkNcnnInCropLayer();
    return layer;
}

IVideoMatting* createVideoMatting() {
    VideoMatting* vm = new VideoMatting();
    return vm;
}

IBaseLayer* createNcnnUploadLayer() {
    VkNcnnUploadLayer* layer = new VkNcnnUploadLayer();
    return layer;
}

}  // namespace aoce