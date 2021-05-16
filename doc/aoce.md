# aoce项目介绍

在[oeip](https://github.com/xxxzhou/oeip)基础上,扩展android平台下功能.

为了方便在多平台编译链接,使用VSCode+CMake.为了方便文档的通用,主要说明文档使用Markdown.

方便导出到别的语言,准备使用[swig](https://www.cnblogs.com/xuruilong100/tag/SWIG%203%20%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8C/).

顺便关注[maui](https://github.com/dotnet/maui/blob/main/README.md)发展.

## 重构计划

aoce_vulkan模块.

1 完成一个类似函数结构图,函数可以包含多个函数,函数与函数对接,用来更方便自动对接执行层,组合层.

2 后期由参数最先决定使用那个Compute shader,由这shader分析输入输出个数,自动创建PipeLayout等各种资料.

3 由参数变动分为三个级别,一是改变输出大小,重置整个graph,二是改变shader/自身buffer,重置自己,三是改变UBO,每桢运行前更新UBO就可.

aoce 模块.

PipeNode类接口不公开,所有接口全隐藏,相应接口通过BaseLayer转接(已完成).

现在一个baselayer只能给一个pipegraph使用,能不能考虑抽离baselayer里的inFormats/outFormats,inLayers/outLayers分给pipenode,做到节点与pipegraph抽离附加一对一关系.

## 导出给用户调用

在框架各模块内部,引用导出的类不要求什么不能用STL,毕竟肯定你编译这些模块肯定是相同编译环境,但是如果导出给别的用户使用,需要限制导出的数据与格式,以保证别的用户与你不同的编译环境也不会有问题.

配合CMake,使用install只导出特殊编写的.h文件,这些文件主要是如下三种作用.

1. C风格的结构,C风格导出帮助函数,与C风格导出用来创建对应工厂/管理对象.

2. 纯净的抽像类,不包含任何STL对象结构,主要用来调用API,用户不要继承这些类.

3. 后缀为Observer的抽像类,用户继承针对接口处理回调.
