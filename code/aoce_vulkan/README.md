# vulkan的基本图元操作

## 功能介绍

## 注意点

每个VkPipeGraph肯定有个数据更新线程,每个层更新的参数可能不在这个数据更新线程上,更新了参数要么导致vkPipeGraph重置,要么置更新UBO flag为true,然后更新UBO在数据更新线程上,这样可以避免很多可能的问题.

VkPipeGraph 提供延迟运行方式，由delayGpu控制,如果为true,则输出结果是上一桢的运行数据生成的，否则当前桢的数据由当前桢数据运行生成。delayGpu为true,CPU不需要在当前桢等待GPU运行结果,不过在android下图像短时间变化较大会有画面割裂的现象.

## 问题

VkPipeGraph在运行时重启时，重启时可能会重置资源造成vulkan device_lost.

(1) 假设因为在运行中，一个线程输出，一个线程运行，这二个线程访问的同一资源造成的？

尝试使用一个中间变量,输出的结果先输出到中间变量上，然后由中间变量输出到显示上，相应尝试结果把输出结果输出到中间变量是会导致上面的device_lost问题（非常奇怪，为什么，明明只是复制下数据），经测试，应该是vkCmdBlitImage这个API导致，换成vkCmdCopyImage可行，二者的区别难道不只是vkCmdBlitImage不需要二个图像一样大？但是还是不行，二个线程，必然要通过同一资源进行交换。

(2) 运行线程重置资源时，GPU里command还在运行相关资源?

大部分设备丢失报错在运行线路上的vkQueueSubmit上，所以猜测可能是这种情况。但是我用非延迟方式运行，理论每桢运行完同步GPU运行，应该不会发生资源重置，还在运行的情况。

(3) 同步二个线程的资源访问。

测试在二个线程tick中使用同步mutex,暂时还没重现这个问题.优化这二者大范围同步，使用VkEvent来确定是否在重置资源中，线程输出检测在重置资源时，不作操作,暂时还没重现出这个问题。

### Resize 重置大小

可以用完全的imageLoad/imageStore实现.

我尝试用sampler来实现,可以大大简化代码,性能提升并不确定,并且上面的方式有优化的空间,结果很奇怪,在运行时看不到结果全黑,但是用RenderDoc查看运行过程时,又能得到正确结果,这里先留个问号.

解决: 1 改texture为texelFetch, 2 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER为VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,奇怪啊,这里应该为sampler才对,是什么导致这种现象,现测试win/android都可以得到结果.
