#pragma once

#if defined __GNUC__
#define AOCE_Func __func__
#elif defined _MSC_VER
#define AOCE_Func __FUNCTION__
#else
#define AOCE_Func ""
#endif

#ifdef _WIN32
#if AOCE_CUDA_EXPORT_DEFINE
#define AOCE_CUDA_EXPORT __declspec(dllexport)
#else
#define AOCE_CUDA_EXPORT __declspec(dllimport)
#endif
#else
#define AOCE_CUDA_EXPORT
#endif

typedef unsigned char uchar;

// opencv相应代码
namespace aoce {
namespace cuda {

#define CUDA_HOST_DEVICE __host__ __device__ __forceinline__

template <typename T>
struct DevPtr {
    typedef T elem_type;
    typedef int index_type;

    enum { elem_size = sizeof(elem_type) };

    T* data;

    CUDA_HOST_DEVICE DevPtr() : data(0) {}
    CUDA_HOST_DEVICE DevPtr(T* data_) : data(data_) {}

    CUDA_HOST_DEVICE size_t elemSize() const { return elem_size; }
    CUDA_HOST_DEVICE operator T*() { return data; }
    CUDA_HOST_DEVICE operator const T*() const { return data; }
};

template <typename T>
struct PtrSz : public DevPtr<T> {
    CUDA_HOST_DEVICE PtrSz() : size(0) {}
    CUDA_HOST_DEVICE PtrSz(T* data_, size_t size_)
        : DevPtr<T>(data_), size(size_) {}

    size_t size;
};

template <typename T>
struct PtrStep : public DevPtr<T> {
    CUDA_HOST_DEVICE PtrStep() : step(0) {}
    CUDA_HOST_DEVICE PtrStep(T* data_, size_t step_)
        : DevPtr<T>(data_), step(step_) {}

    size_t step;

    CUDA_HOST_DEVICE T* ptr(int y = 0) {
        return (T*)((char*)DevPtr<T>::data + y * step);
    }
    CUDA_HOST_DEVICE const T* ptr(int y = 0) const {
        return (const T*)((const char*)DevPtr<T>::data + y * step);
    }

    CUDA_HOST_DEVICE T& operator()(int y, int x) { return ptr(y)[x]; }
    CUDA_HOST_DEVICE const T& operator()(int y, int x) const {
        return ptr(y)[x];
    }
};

template <typename T>
struct PtrStepSz : public PtrStep<T> {
    CUDA_HOST_DEVICE PtrStepSz() : width(0), height(0) {}
    CUDA_HOST_DEVICE PtrStepSz(int height_, int width_, T* data_, size_t step_)
        : PtrStep<T>(data_, step_), width(width_), height(height_) {}

    template <typename U>
    explicit PtrStepSz(const PtrStepSz<U>& d)
        : PtrStep<T>((T*)d.data, d.step), width(d.width), height(d.height) {}

    int width;   // cols;
    int height;  // rows;
};

}  // namespace cuda
}  // namespace aoce