# GLSL单独分离出来

[glslangValidator](http://manpages.ubuntu.com/manpages/focal/en/man1/glslangValidator.1.html)

考虑可能有100多种效果,glsl对应不同条件生成可能超过100多个文件,所以建立一个单独文件夹管理所有glsl文件.

glslindex.txt 文件用来指示如何需要编译的glsl文件目录.

compileglsl.py 用来针对glslindex.txt编译glsl文件.

## opengl/cuda computer shader 线程

gl_NumWorkGroups/gridDim: 所有线程块的多少.
gl_WorkGroupSize/blockDim: 本身线程块的大小.
gl_WorkGroupID/blockIdx: 线程块在所有线程块中索引.
gl_LocalInvocationID/threadIdx: 线程在线程块中的索引.
gl_GlobalInvocationID = blockIdx*blockDim + gl_LocalInvocationID

## 文档

[Ray tracing with OpenGL Compute Shaders (Part I)](https://github.com/LWJGL/lwjgl3-wiki/wiki/2.6.1.-Ray-tracing-with-OpenGL-Compute-Shaders-%28Part-I%29)

[The Possibilities of Compute Shaders](https://kola.opus.hbz-nrw.de/opus45-kola/frontdoor/deliver/index/docId/786/file/JochenHunzBachelorThesis.pdf)

[Compute_Shader](https://www.khronos.org/opengl/wiki/Compute_Shader)

[Atomic_Counter](https://www.khronos.org/opengl/wiki/Atomic_Counter)

[VulkanSubgroups(vulkan subgroups example for reduce and scan)](https://github.com/mmaldacker/VulkanSubgroups/blob/master/Reduce.comp) (#extension GL_KHR_shader_subgroup_arithmetic : enable)

[Parallel reduce and scan on the GPU](https://cachemiss.xyz/blog/parallel-reduce-and-scan-on-the-GPU)

[使用计算着色器计算平均亮度](https://therealmjp.github.io/posts/average-luminance-compute-shader/)
