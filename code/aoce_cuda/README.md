# aoce_cuda 提供CUDA的aoce封装实现

## 主要改进

在oeip上使用了opencv的cuda模块来完成一些功能,aoce中去掉这些联系.

主要仿opencv里cuda模块的gpumat实现CudaMat/Operate,来方便C++/CUDA数据的传递.
