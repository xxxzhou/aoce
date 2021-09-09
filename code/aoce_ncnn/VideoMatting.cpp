#include "VideoMatting.hpp"

namespace aoce {

VideoMatting::VideoMatting(/* args */) {
    net = std::make_unique<ncnn::Net>();
    net->opt.use_vulkan_compute = true;
    netFormet.width = 512;
    netFormet.height = 512;
    netFormet.imageType = ImageType::bgr8;
}

VideoMatting::~VideoMatting() {
    // if (net) {
    //     const ncnn::VulkanDevice* device = net->vulkan_device();
    //     device->reclaim_blob_allocator(blobVkallocator);
    //     device->reclaim_staging_allocator(stagingVkallocator);
    // }
}

bool VideoMatting::initNet(IBaseLayer* ncnnInLayer,
                           IBaseLayer* ncnnUploadLayer) {
    if (bInitNet) {
        return true;
    }
    std::string paramFile = "net/rvm-512.param";
    std::string binFile = "net/rvm-512.bin";
    int32_t ret = loadNet(net.get(), paramFile, binFile);
    if (ret == 0) {
        ncnnLayer = static_cast<VkNcnnInCropLayer*>(ncnnInLayer);
        if (!ncnnLayer) {
            logMessage(
                LogLevel::warn,
                "VideoMatting initialize the network parameters ncnnlayer, "
                "please use the function createNcnnInLayer to create");
            return false;
        }
        uploadLayer = static_cast<VkNcnnUploadLayer*>(ncnnUploadLayer);
        if (!uploadLayer) {
            logMessage(
                LogLevel::warn,
                "VideoMatting initialize the network parameters "
                "ncnnUploadLayer, "
                "please use the function createNcnnUploadLayer to create");
            return false;
        }
        NcnnInParamet paramet = {};
        paramet.mean = {123.675f, 116.28f, 103.53f, 0.0f};
        paramet.scale = {0.01712475f, 0.0175f, 0.01742919f, 1.0f};
        paramet.outWidth = netFormet.width;
        paramet.outHeight = netFormet.height;
        ncnnLayer->updateParamet(paramet);
        ncnnLayer->setObserver(this, netFormet.imageType);
        uploadLayer->setImageFormat(netFormet);

        temp1 = ncnn::Mat(64, 64, 16);
        temp2 = ncnn::Mat(32, 32, 20);
        temp3 = ncnn::Mat(16, 16, 40);
        temp4 = ncnn::Mat(8, 8, 64);

        temp1.fill(0.0f);
        temp2.fill(0.0f);
        temp3.fill(0.0f);
        temp4.fill(0.0f);

        bInitNet = true;
    }
    return bInitNet;
}

void VideoMatting::onResult(VulkanBuffer* buffer,
                            const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();

    // 得到当前区域
    ncnn::Extractor netEx = net->create_extractor();
    ncnn::Mat inMat(netFormet.width, netFormet.height, 3, buffer->getCpuData());

    netEx.input("src", inMat);
    netEx.input("r1i", temp1);
    netEx.input("r2i", temp2);
    netEx.input("r3i", temp3);
    netEx.input("r4i", temp4);

    ncnn::Mat pha;
    ncnn::Mat out1, out2, out3, out4;
    netEx.extract("r4o", out4);
    netEx.extract("r3o", out3);
    netEx.extract("r2o", out2);
    netEx.extract("r1o", out1);

    netEx.extract("pha", pha);

    temp1 = out1.clone();
    temp2 = out2.clone();
    temp3 = out3.clone();
    temp4 = out4.clone();

    uploadLayer->uploadBuffer(pha.data);

    long long time2 = getNowTimeStamp();
    std::string tmsg;
    string_format(tmsg, "video matting time:", time2 - time1);
    logMessage(LogLevel::info, tmsg);
}

}  // namespace aoce