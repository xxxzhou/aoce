#include "CudaMat.hpp"

void checkCudaError(cudaError_t err, const char* file, const int line,
                    const char* func) {
    if (cudaSuccess != err) {
        std::string msg;
        aoce::string_format(msg, "cuda error: ", cudaGetErrorString(err),
                      " file: ", file, " line: ", line, " in function: ", func);
        logMessage(aoce::LogLevel::error, msg);
    }
}

namespace aoce {
namespace cuda {

CudaMat::CudaMat(/* args */) {}

CudaMat::~CudaMat() { release(); }

int32_t CudaMat::elemSize() { return AOCE_CV_ELEM_SIZE(flags); }
int32_t CudaMat::elemSize1() { return AOCE_CV_ELEM_SIZE1(flags); }
int32_t CudaMat::channels() { return AOCE_CV_MAT_CN(flags); }

bool CudaMat::empty() { return data == nullptr; }

void CudaMat::release() {
    if (data) {
        cudaFree(data);
        data = nullptr;
        step = width = height = 0;
    }
}

void CudaMat::create(int32_t pwidth, int32_t pheight, int32_t pflags) {
    assert(pwidth >= 0 && pheight >= 0);
    if (width == pwidth && height == pheight && flags == pflags && data) {
        return;
    }
    if (data) {
        release();
    }
    if (pwidth > 0 && pheight > 0) {
        width = pwidth;
        height = pheight;
        flags = pflags;
        int esz = elemSize();
        if (width > 1 && height > 1) {
            AOCE_CUDEV_SAFE_CALL(
                cudaMallocPitch((void**)&data, &step, esz * width, height));
        } else {
            AOCE_CUDEV_SAFE_CALL(
                cudaMalloc((void**)&data, esz * width * height));
            step = esz * width;
        }
        continuons = (step == esz * width);
    }
}

void CudaMat::upload(uint8_t* pdata, int32_t pstep, cudaStream_t stream) {
    int32_t cstep = pstep == 0 ? (elemSize() * width) : pstep;
    if (stream) {
        cudaMemcpy2DAsync(data, step, pdata, cstep, width * elemSize(), height,
                          cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy2D(data, step, pdata, cstep, width * elemSize(), height,
                     cudaMemcpyHostToDevice);
    }
}

void CudaMat::download(uint8_t* pdata, int32_t pstep, cudaStream_t stream) {
    int32_t cstep = pstep == 0 ? (elemSize() * width) : pstep;
    if (stream) {
        cudaMemcpy2DAsync(pdata, cstep, data, step, width * elemSize(), height,
                          cudaMemcpyDeviceToHost, stream);
    } else {
        cudaMemcpy2D(pdata, cstep, data, step, width * elemSize(), height,
                     cudaMemcpyDeviceToHost);
    }
}

void CudaMat::copyTo(CudaMatRef mat, cudaStream_t stream) {
    mat->create(width, height, flags);
    if (stream) {
        cudaMemcpy2DAsync(mat->data, mat->step, data, step, width * elemSize(),
                          height, cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2D(mat->data, mat->step, data, step, width * elemSize(),
                     height, cudaMemcpyDeviceToDevice);
    }
}

uint8_t* CudaMat::ptr(int32_t y) {
    assert(y < height);
    return data + step * y;
}

int32_t CudaMat::getWidth() { return width; }

int32_t CudaMat::getHeight() { return height; }

size_t CudaMat::getStep() { return step; }

}  // namespace cuda
}  // namespace aoce