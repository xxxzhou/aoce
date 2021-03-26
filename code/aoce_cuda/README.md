# aoce_cuda 提供CUDA的aoce封装实现

## 主要改进

opencv的cuda模块首先太大了,后续发展不确定,因此在aoce上去掉opencv相关的编译链接.

主要仿opencv里cuda模块的gpumat实现CudaMat/Operate,来方便C++/CUDA数据的传递,再次模仿opencv cuda模块来实现一些常用操作.
