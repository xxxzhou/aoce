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
};

// 在外部第三方插件,可以直接提供new XXXLayer(),里面注明对应的gpu类型就行
}