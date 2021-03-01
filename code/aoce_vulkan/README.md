# vulkan的基本图元操作

## 功能介绍

(1) android vulkan直接与opengl绑定。

(2) windows 如果vulkan与dx11交互完成，后续可以考虑放弃dx11相关模块开发

PS:

[dxgi_interop](https://github.com/krOoze/Hello_Triangle/blob/dxgi_interop/src/WSI/DxgiWsi.h)

[BindImageMemory](https://github.com/roman380/VulkanSdkDemos/blob/d3d11-image-interop/BindImageMemory2/BindImageMemory2.cpp#L154)

[dx11-vulkan-keymutex](https://github.com/KhronosGroup/VK-GL-CTS/blob/master/external/vulkancts/modules/vulkan/synchronization/vktSynchronizationWin32KeyedMutexTests.cpp)

## 注意点

每个VkPipeGraph肯定有个数据更新线程,每个层更新的参数可能不在这个数据更新线程上,更新了参数要么导致vkPipeGraph重置,要么置更新UBO flag为true,然后更新UBO在数据更新线程上,这样可以避免很多可能的问题.

VkPipeGraph 提供延迟运行方式，由delayGpu控制,如果为true,则输出结果是上一桢的运行数据生成的，否则当前桢的数据由当前桢数据运行生成。delayGpu为true,CPU不需要在当前桢等待GPU运行结果,不过在android下图像短时间变化较大会有画面割裂的现象.

## 问题

1 VkPipeGraph在运行时重启时，重启时会重置资源可能造成vulkan device_lost.

(1) 假设因为在运行中，一个线程输出，一个线程运行，这二个线程访问的同一资源造成的？

尝试使用一个中间变量,输出的结果先输出到中间变量上，然后由中间变量输出到显示上，相应尝试结果把输出结果输出到中间变量是会导致上面的device_lost问题（非常奇怪，为什么，明明只是复制下数据），经测试，应该是vkCmdBlitImage这个API导致，换成vkCmdCopyImage可行，二者的区别难道不只是vkCmdBlitImage不需要二个图像一样大？但是还是不行，二个线程，必然要通过同一资源进行交换。

测试在二个线程tick中使用同步mutex,暂时还没重现这个问题.优化这二者大范围同步，使用VkEvent来确定是否在重置资源中，线程输出检测在重置资源时，不作操作,这个操作后还没重现出这个问题。

2 DX11 与 Vulkan交互蓝屏与无显示？

(1) VK submit提交渲染管线时，有交互很容易导致crash,机器蓝屏，猜想原因是DX11渲染与VK写入对接的keymute需要满足一要求一释放，而cuda/dx11可以自己控制。

先测试下，要保证一要求一释放，又要dx11与vulkan不在同一线程与同一频率，只能加入一个中间dx11纹理，测试看看效果。

我晕，导致crash主要是因为vkCmdBlitImage导致的，用vkCmdCopyImage无问题。

(2) 新问题,选择移动窗口会导致vulkan timeout.

嗯，这个问题用上面的方法解决，加入中间层，原因应该就是如上需要满足计算与使用二线程无等待关系，那这样，直接用中间层，VK运行线程直接写入结果，也不需要用D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX，改为D3D11_RESOURCE_MISC_SHARED，用vkFence保证GPU里VK CommandBuffer执行顺序与DX11拷贝到临时变量的顺序。这样VK执行线程与DX11外部显示线程可以用不同频率跑。

### Resize 重置大小

可以用完全的imageLoad/imageStore实现.

我尝试用sampler来实现,可以大大简化代码,性能提升并不确定,并且上面的方式有优化的空间,结果很奇怪,在运行时看不到结果全黑,但是用RenderDoc查看运行过程时,又能得到正确结果,这里先留个问号.

解决: 1 改texture为texelFetch, 2 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER为VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,奇怪啊,这里应该为sampler才对,是什么导致这种现象,现测试win/android都可以得到结果.
