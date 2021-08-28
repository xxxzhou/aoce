# ncnn源码解析

## ncnn层通过CMake配置

宏NCNN_STRING 表示层用一个str标识,否则只用int索引表示.

卷积神经网络平常过久就有人提出新的思路,一般来说,对应一些新的运行层,所以新增层对源码改动量是一个比较大的考量.

宏DEFINE_LAYER_CREATOR(name): 声明根据对应name创建对应层的函数指针,与头文件layer_registry.h对应起来.

通过CMake动态生成layer_declaration.h,这个文件里有所有层的工厂类,根据条件生成avx/avx2/vulkan对应层,其中DEFINE_LAYER_CREATOR指明layer_registry.h里每个name的函数指针里的实现.

外部扩展层:register_custom_layer(const char* type, layer_creator_func creator);与如上对应,type/createor表示name/layer,具体可以参考YoloV5Focus相关.

## param文件

前二位应该表示输入/输出个数.一般情况下,然后根据前面二位解析输入/输出层名字.

卷积层里: 1/11表示kernel w/h,2/12表示dilation w/h(空洞卷积),3/13表示stride w/h,4/14表示pad w/h.5表示是否有偏置,6卷积核参数大小 in(x)out(x)ksize.
