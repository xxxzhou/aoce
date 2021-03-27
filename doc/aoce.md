# aoce项目介绍

在[oeip](https://github.com/xxxzhou/oeip)基础上,扩展android平台下功能.

为了方便在多平台编译链接,使用VSCode+CMake.为了方便文档的通用，主要说明文档使用Markdown.

方便导出到别的语言，准备使用[swig](https://www.cnblogs.com/xuruilong100/tag/SWIG%203%20%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8C/).

顺便关注[maui](https://github.com/dotnet/maui/blob/main/README.md)发展。

## 重构计划

aoce_vulkan模块.

1 完成一个类似函数结构图,函数可以包含多个函数,函数与函数对接,用来更方便自动对接执行层,组合层.

2 后期由参数最先决定使用那个Compute shader,由这shader分析输入输出个数,自动创建PipeLayout等各种资料.
