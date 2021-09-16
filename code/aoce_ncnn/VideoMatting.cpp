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
#if VULKAN_OUTPUT
    const ncnn::VulkanDevice* device = getNgParamet().vkDevice;
    device->reclaim_blob_allocator(blobVkallocator);
    device->reclaim_staging_allocator(stagingVkallocator);
#endif
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
#if VULKAN_OUTPUT
        vkCmd = std::make_unique<VkCommand>();
        blob_vkallocator = getNgParamet().vkDevice->acquire_blob_allocator();
        staging_vkallocator =
            getNgParamet().vkDevice->acquire_staging_allocator();
        net->opt.blob_vkallocator = blob_vkallocator;
        net->opt.workspace_vkallocator = blob_vkallocator;
        net->opt.staging_vkallocator = staging_vkallocator;
        uploadLayer->setImageFormat(netFormet, net->opt.use_fp16_storage);
        ncnnLayer->updateParamet(paramet, net->opt.use_fp16_storage);
#else
        ncnnLayer->updateParamet(paramet, false);
        uploadLayer->setImageFormat(netFormet, false);
#endif
        ncnnLayer->setObserver(this, netFormet.imageType);
#if VULKAN_OUTPUT
        ncnn::VkAllocator* vkAllocator = getNgParamet().vkAllocator;
        // 输入Mat的通道数如果是4的倍数,那么Mat对应的VkMat通道数/4,elempack*4.
        // 详细请看VkCompute::record_upload,对应思路不明(CPU对应SIMD,GPU了).
        uint32_t elemsize = net->opt.use_fp16_storage ? 2u : 4u;
        temp1.create(64, 64, 4, 4 * elemsize, 4, vkAllocator);
        temp2.create(32, 32, 5, 4 * elemsize, 4, vkAllocator);
        temp3.create(16, 16, 10, 4 * elemsize, 4, vkAllocator);
        temp4.create(8, 8, 16, 4 * elemsize, 4, vkAllocator);
        // temp1.create(64, 64, 16, elemsize, vkAllocator);
        // temp2.create(32, 32, 20, elemsize, vkAllocator);
        // temp3.create(16, 16, 40, elemsize, vkAllocator);
        // temp4.create(8, 8, 64, elemsize, vkAllocator);
        vkCmd->fill(temp1.buffer(), temp1.buffer_capacity(), 0, 0);
        vkCmd->fill(temp2.buffer(), temp2.buffer_capacity(), 0, 0);
        vkCmd->fill(temp3.buffer(), temp3.buffer_capacity(), 0, 0);
        vkCmd->fill(temp4.buffer(), temp4.buffer_capacity(), 0, 0);
        vkCmd->submit();
        vkCmd->reset();
#else
        temp1 = ncnn::Mat(64, 64, 16);
        temp2 = ncnn::Mat(32, 32, 20);
        temp3 = ncnn::Mat(16, 16, 40);
        temp4 = ncnn::Mat(8, 8, 64);

        temp1.fill(0.0f);
        temp2.fill(0.0f);
        temp3.fill(0.0f);
        temp4.fill(0.0f);
#endif
        bInitNet = true;
    }
    return bInitNet;
}

void VideoMatting::onResult(ncnn::VkMat& vkMat,
                            const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();

    // 得到当前区域
    ncnn::Extractor netEx = net->create_extractor();
// ncnn::Mat inMat(netFormet.width, netFormet.height, 3,
// buffer->getCpuData());
#if VULKAN_OUTPUT
    netEx.input("src", vkMat);
#else
    netEx.input("src", vkMat.mapped());
#endif
    netEx.input("r1i", temp1);
    netEx.input("r2i", temp2);
    netEx.input("r3i", temp3);
    netEx.input("r4i", temp4);

#if VULKAN_OUTPUT
    ncnn::VkMat vkPha;
    ncnn::VkMat out1, out2, out3, out4;
    ncnn::VkCompute cmd(getNgParamet().vkDevice);
    netEx.extract("r4o", out4, cmd);
    netEx.extract("r3o", out3, cmd);
    netEx.extract("r2o", out2, cmd);
    netEx.extract("r1o", out1, cmd);
    netEx.extract("pha", vkPha, cmd);
    // 等待执行完成
    cmd.submit_and_wait();
    VulkanBuffer* vbBuffer = uploadLayer->getVkBuffer();
    // 复制
    vkCmd->record(vkPha.buffer(), vbBuffer->buffer, 0,
                  vbBuffer->getBufferSize());
    vkCmd->record(out4.buffer(), temp4.buffer(), 0, out4.buffer_capacity());
    vkCmd->record(out3.buffer(), temp3.buffer(), 0, out3.buffer_capacity());
    vkCmd->record(out2.buffer(), temp2.buffer(), 0, out2.buffer_capacity());
    vkCmd->record(out1.buffer(), temp1.buffer(), 0, out1.buffer_capacity());
    vkCmd->submit();
    vkCmd->reset();
#else
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
#endif
    long long time2 = getNowTimeStamp();
    total += (time2 - time1);
    totalIndex++;
    std::string tmsg;
    string_format(tmsg, "video matting time:", time2 - time1,
                  " avg time:", total / totalIndex);
    logMessage(LogLevel::info, tmsg);
}

}  // namespace aoce