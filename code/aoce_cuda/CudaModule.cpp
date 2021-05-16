#include "cudaModule.hpp"

#include <AoceManager.hpp>

#include "CuLayerFactory.hpp"
#include "layer/CuPipeGraph.hpp"

namespace aoce {
namespace cuda {

CudaModule::CudaModule(/* args */) {}

CudaModule::~CudaModule() {}

bool CudaModule::loadModule() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (count <= 0) {
        logMessage(LogLevel::warn, "cuda not find gpu device.");
        return false;
    }
    int i = 0;
    for (i = 0; i < count; ++i) {
        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            // 选用第一个独立显卡,没有就不管了
            if (prop.integrated == 0) {
                std::string msg;
                string_format(msg, "select device: ", prop.name,
                              " compute capability: ", prop.major, prop.minor);
                logMessage(LogLevel::info, msg);
                cudaSetDevice(i);
                break;
            }
        }
    }

    logMessage(LogLevel::info, "add cuda factory.");
    AoceManager::Get().addPipeGraphFactory(GpuType::cuda,
                                           new CuPipeGraphFactory());
    AoceManager::Get().addLayerFactory(GpuType::cuda, new CuLayerFactory());
    return true;
}

void CudaModule::unloadModule() {
    logMessage(LogLevel::info, "remove cuda factory.");
    AoceManager::Get().removePipeGraphFactory(GpuType::cuda);
    AoceManager::Get().removeLayerFactory(GpuType::cuda);
}

ADD_MODULE(CudaModule, aoce_cuda)

}  // namespace cuda
}  // namespace aoce