# Vulkan glsl

## Vk_format与glsl对应

[spirvenv.txt](https://git.sr.ht/~sircmpwn/vulkan-docs/tree/c75ae4fb18742c564d29503b598413879f920d2f/item/appendices/spirvenv.txt)

.SPIR-V and Vulkan Image Format Compatibility
[cols="2*", options="header"]
|====
|SPIR-V Image Format    |Compatible Vulkan Format
|code:Rgba32f           |ename:VK_FORMAT_R32G32B32A32_SFLOAT
|code:Rgba16f           |ename:VK_FORMAT_R16G16B16A16_SFLOAT
|code:R32f              |ename:VK_FORMAT_R32_SFLOAT
|code:Rgba8             |ename:VK_FORMAT_R8G8B8A8_UNORM
|code:Rgba8Snorm        |ename:VK_FORMAT_R8G8B8A8_SNORM
|code:Rg32f             |ename:VK_FORMAT_R32G32_SFLOAT
|code:Rg16f             |ename:VK_FORMAT_R16G16_SFLOAT
|code:R11fG11fB10f      |ename:VK_FORMAT_B10G11R11_UFLOAT_PACK32
|code:R16f              |ename:VK_FORMAT_R16_SFLOAT
|code:Rgba16            |ename:VK_FORMAT_R16G16B16A16_UNORM
|code:Rgb10A2           |ename:VK_FORMAT_A2B10G10R10_UNORM_PACK32
|code:Rg16              |ename:VK_FORMAT_R16G16_UNORM
|code:Rg8               |ename:VK_FORMAT_R8G8_UNORM
|code:R16               |ename:VK_FORMAT_R16_UNORM
|code:R8                |ename:VK_FORMAT_R8_UNORM
|code:Rgba16Snorm       |ename:VK_FORMAT_R16G16B16A16_SNORM
|code:Rg16Snorm         |ename:VK_FORMAT_R16G16_SNORM
|code:Rg8Snorm          |ename:VK_FORMAT_R8G8_SNORM
|code:R16Snorm          |ename:VK_FORMAT_R16_SNORM
|code:R8Snorm           |ename:VK_FORMAT_R8_SNORM
|code:Rgba32i           |ename:VK_FORMAT_R32G32B32A32_SINT
|code:Rgba16i           |ename:VK_FORMAT_R16G16B16A16_SINT
|code:Rgba8i            |ename:VK_FORMAT_R8G8B8A8_SINT
|code:R32i              |ename:VK_FORMAT_R32_SINT
|code:Rg32i             |ename:VK_FORMAT_R32G32_SINT
|code:Rg16i             |ename:VK_FORMAT_R16G16_SINT
|code:Rg8i              |ename:VK_FORMAT_R8G8_SINT
|code:R16i              |ename:VK_FORMAT_R16_SINT
|code:R8i               |ename:VK_FORMAT_R8_SINT
|code:Rgba32ui          |ename:VK_FORMAT_R32G32B32A32_UINT
|code:Rgba16ui          |ename:VK_FORMAT_R16G16B16A16_UINT
|code:Rgba8ui           |ename:VK_FORMAT_R8G8B8A8_UINT
|code:R32ui             |ename:VK_FORMAT_R32_UINT
|code:Rgb10a2ui         |ename:VK_FORMAT_A2B10G10R10_UINT_PACK32
|code:Rg32ui            |ename:VK_FORMAT_R32G32_UINT
|code:Rg16ui            |ename:VK_FORMAT_R16G16_UINT
|code:Rg8ui             |ename:VK_FORMAT_R8G8_UINT
|code:R16ui             |ename:VK_FORMAT_R16_UINT
|code:R8ui              |ename:VK_FORMAT_R8_UINT
|====
