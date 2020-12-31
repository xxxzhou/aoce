#pragma once

// 仿opencv里的GpuMat
#include <vector_functions.h>

#include <Aoce.hpp>
#include <memory>

#include "cuda/CudaTypes.hpp"

__host__ __forceinline__ void checkCudaError(cudaError_t err, const char* file,
                                             const int line, const char* func) {
    if (cudaSuccess != err) {
        std::string msg;
        string_format(msg, "cuda error: ", cudaGetErrorString(err),
                      " file: ", file, " line: ", line, " in function: ", func);
        logMessage(aoce::LogLevel::error, msg);
    }
}

#define AOCE_CUDEV_SAFE_CALL(expr) \
    checkCudaError((expr), __FILE__, __LINE__, AOCE_Func)

namespace aoce {
namespace cuda {

typedef std::shared_ptr<class CudaMat> CudaMatRef;

class AOCE_CUDA_EXPORT CudaMat {
   private:
    /* data */
    uint8_t* data = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    size_t step = 0;
    int32_t flags = 0;
    bool continuons = false;

   public:
    CudaMat(/* args */);
    ~CudaMat();

   public:
    // returns element size in bytes
    int32_t elemSize();
    // returns the size of element channel in bytes
    int32_t elemSize1();
    int32_t channels();
    bool empty();

    void release();
    void create(int32_t width, int32_t height, int32_t flags);

    void upload(uint8_t* data, int32_t step = 0, cudaStream_t stream = nullptr);
    void download(uint8_t* data, int32_t step = 0,
                  cudaStream_t stream = nullptr);

    void copyTo(CudaMatRef mat, cudaStream_t stream = nullptr);

    uint8_t* ptr(int32_t y = 0);
    int32_t getWidth();
    int32_t getHeight();
    size_t getStep();

   public:
    template <typename _Tp>
    _Tp* ptr(int y = 0) {
        return (_Tp*)ptr(y);
    }
    template <typename _Tp>
    const _Tp* ptr(int y = 0) const {
        return (const _Tp*)ptr(y);
    };
    template <typename _Tp>
    operator PtrStepSz<_Tp>() const {
        return PtrStepSz<_Tp>(height, width, (_Tp*)data, step);
    };
    template <typename _Tp>
    operator PtrStep<_Tp>() const {
        return PtrStep<_Tp>((_Tp*)data, step);
    };
};

}  // namespace cuda
}  // namespace aoce
