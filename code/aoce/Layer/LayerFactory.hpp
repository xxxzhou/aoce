#pragma once

#include "BaseLayer.hpp"
#include "InputLayer.hpp"
#include "OutputLayer.hpp"
namespace aoce {

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class ACOE_EXPORT LayerFactory {
   public:
    LayerFactory(){};
    virtual ~LayerFactory(){};

   public:
    virtual InputLayer* crateInput() = 0;
    virtual OutputLayer* createOutput() = 0;
    virtual YUV2RGBALayer* createYUV2RGBA() = 0;
    virtual RGBA2YUVLayer* createRGBA2YUV() = 0;
    virtual TexOperateLayer* createTexOperate() = 0;
    virtual TransposeLayer* createTranspose() = 0;
    virtual ReSizeLayer* createSize() = 0;
    virtual BlendLayer* createBlend() = 0;
};

// 在外部第三方插件,可以直接提供new XXXLayer(),里面注明对应的gpu类型就行
// 以及这个层对应更新接口
}