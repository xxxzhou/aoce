#pragma once

#include "AoceVkExtra.hpp"

// 类注释主要来源 https://gitee.com/xudoubi/GPUImage

namespace aoce {
namespace vulkan {

// 色彩调整
#pragma region Color adjustments
// 亮度:调整后的亮度(-1.0-1.0,默认值为0.0)
AOCE_VE_EXPORT ITLayer<float>* createBrightnessLayer();
// 曝光度:调整后的曝光度(-10.0-10.0,默认值为0.0)
AOCE_VE_EXPORT ITLayer<float>* createExposureLayer();
// 对比度:调整后的对比度(0.0-4.0,默认值为1.0)
AOCE_VE_EXPORT ITLayer<float>* createContrastLayer();
// 饱和度:应用于图像的饱和度或去饱和度(0.0-2.0,默认值为1.0)
AOCE_VE_EXPORT ITLayer<float>* createSaturationLayer();
// gamma:要应用的gamma调整(0.0-3.0,默认值为1.0)
AOCE_VE_EXPORT ITLayer<float>* createGammaLayer();
// 亮度高于阈值的像素将反转其颜色
AOCE_VE_EXPORT ITLayer<float>* createSolarizeLayer();
// 类似Photoshop的色阶调整,范围从0.0到1.0,默认值为0.5
AOCE_VE_EXPORT ITLayer<LevelsParamet>* createLevelsLayer();
// 通过将矩阵应用于图像来变换图像的颜色
AOCE_VE_EXPORT ITLayer<ColorMatrixParamet>* createColorMatrixLayer();
// 调整图像的各个RGB通道,每个通道范围0.0-1.0
AOCE_VE_EXPORT ITLayer<vec3>* createRGBLayer();
// 调整图像的色调,以度为单位.默认为90度
AOCE_VE_EXPORT ITLayer<float>* createHueLayer();
// 调整图像的鲜艳度,默认设置为0.0,建议的最小/最大值分别为-1.2和1.2.
AOCE_VE_EXPORT ITLayer<float>* createVibranceLayer();
// 调整图像的白平衡.
AOCE_VE_EXPORT ITLayer<WhiteBalanceParamet>* createBalanceLayer();
// 基于样条曲线为每个颜色通道调整图像的颜色
// GPUImageToneCurveFilter 后面移植

// 调整图像的阴影和高光
AOCE_VE_EXPORT ITLayer<HighlightShadowParamet>* createHighlightShadowLayer();
// 使用颜色和强度独立地着色图像的阴影和高光
AOCE_VE_EXPORT ITLayer<HighlightShadowTintParamet>*
createHighlightShadowTintLayer();
// 使用RGB颜色查找图像来重新映射图像中的颜色.
// 在graph run之前,使用loadLookUp加载映射图里的数据
// Amatorka/MissEtikate
AOCE_VE_EXPORT LookupLayer* createLookupLayer();
// 基于双重查找的颜色重新映射滤镜
// loadLookUp1/loadLookUp2分别加载对应映射表里的数据
AOCE_VE_EXPORT SoftEleganceLayer* createSoftEleganceLayer();
// 肤色调整滤镜,可影响浅肤色颜色的唯一范围,并相应地调整粉红色/绿色或粉红色/橙色范围.
// 默认值针对白皙的皮肤,但可以根据需要进行调整.
AOCE_VE_EXPORT ITLayer<SkinToneParamet>* createSkinToneLayer();
// 反转图像的颜色
AOCE_VE_EXPORT BaseLayer* createColorInvertLayer();
// 将图像转换为灰度
AOCE_VE_EXPORT BaseLayer* createLuminanceLayer();
// 根据每个像素的亮度将图像转换为单色版本
AOCE_VE_EXPORT ITLayer<MonochromeParamet>* createMonochromeLayer();
// 根据图像的亮度在两种用户指定的颜色之间进行混合
AOCE_VE_EXPORT ITLayer<FalseColorParamet>* createFalseColorLayer();
// 用于添加或删除雾度(类似于UV滤镜
AOCE_VE_EXPORT ITLayer<HazeParamet>* createHazeLayer();
// 棕褐色调滤镜,参数表示棕褐色替换正常图像颜色的程度,范围0.0-1.0,默认1.0
AOCE_VE_EXPORT ITLayer<float>* createSepiaLayer();
// 调整传入图像的Alpha通道,将每个像素的输入Alpha通道乘以(0.0-1.0,默认值为1.0)的值
AOCE_VE_EXPORT ITLayer<float>* createOpacityLayer();
// GPUImageSolidColorGenerator 直接由VkLayer::clearColor实现.

// 亮度高于阈值的像素将显示为白色,而低于阈值的像素将为黑色,范围0.0-1.0,默认0.5
AOCE_VE_EXPORT ITLayer<float>* createLuminanceThresholdLayer();
// 确定像素周围的局部亮度,如果像素低于该局部亮度,则将其变为黑色,如果高于该像素,则将其变为白色.
// 这对于在变化的光照条件下挑选文本很有用.
AOCE_VE_EXPORT ITLayer<AdaptiveThresholdParamet>*
createAdaptiveThresholdLayer();
// 这将应用阈值操作,其中将根据场景的平均亮度连续调整阈值.
AOCE_VE_EXPORT ITLayer<float>* createAverageLuminanceThresholdLayer();
// 这将分析传入的图像并创建一个输出直方图,其输出每种颜色值的频率.
// 该滤波器根据bSingle,输出[1x256]或是[4x256]的图像
AOCE_VE_EXPORT BaseLayer* createHistogramLayer(bool bSingle = true);
// GPUImageAverageColor/GPUImageLuminosity由VkReduceLayer替代,用于求整张图的颜色和,最大值,最小值

// 色度键扣像 参考
// https://www.unrealengine.com/en-US/tech-blog/setting-up-a-chroma-key-material-in-ue4?sessionInvalidated=true
AOCE_VE_EXPORT ITLayer<ChromaKeyParamet>* createChromaKeyLayer();
// 用来添加色相旋转/饱和度/亮度调整
AOCE_VE_EXPORT HSBLayer* createHSBLayer();
#pragma endregion

// 图像处理
#pragma region Image processing
// 任意2D或3D变换应用于图像
// GPUImageTransformFilter 后面移植

// 裁剪图像的特定区域,然后仅将该区域传递到滤镜的下一个阶段
AOCE_VE_EXPORT ITLayer<CropParamet>* createCropLayer();
// 使用Lanczos重采样对图像进行上采样或下采样,质量明显优于标准线性,后面移植
// GPUImageLanczosResamplingFilter

// 锐化图像,要应用的清晰度调整(-4.0-4.0,默认值为0.0)
AOCE_VE_EXPORT ITLayer<SharpenParamet>* createSharpenLayer();
// 应用不清晰的蒙版
AOCE_VE_EXPORT ITLayer<UnsharpMaskParamet>* createUnsharpMaskLayer();
// 高斯模糊,参考opencv cuda模块实现,现支持RGBA8/R8/RGBA32F
AOCE_VE_EXPORT ITLayer<GaussianBlurParamet>* createGaussianBlurLayer(
    ImageType imageType = ImageType::rgba8);
// Box模糊,参考上面高斯模糊卷积分离优化,现支持RGBA8/R8/RGBA32F
AOCE_VE_EXPORT ITLayer<KernelSizeParamet>* createBoxFilterLayer(
    ImageType imageType = ImageType::rgba8);
// 高斯模糊,在特定圆形区域不模糊
AOCE_VE_EXPORT ITLayer<BlurSelectiveParamet>* createBlurSelectiveLayer();
// 只在特定圆形区域高斯模糊
AOCE_VE_EXPORT ITLayer<BulrPositionParamet>* createBlurPositionLayer();
// 应用IOSBlur
AOCE_VE_EXPORT ITLayer<IOSBlurParamet>* createIOSBlurLayer();
// 不定长区域的中值模糊,不推荐,慢
AOCE_VE_EXPORT ITLayer<uint32_t>* createMedianLayer(bool bSingle = true);
// 3x3区域的中值模糊,实时可用
AOCE_VE_EXPORT BaseLayer* createMedianK3Layer(bool bSingle = true);
// 双边滤波,保留锐利边缘的同时模糊相似的颜色值
AOCE_VE_EXPORT ITLayer<BilateralParamet>* createBilateralLayer();
// 模拟倾斜移位镜头效果
AOCE_VE_EXPORT ITLayer<TiltShiftParamet>* createTiltShiftLayer();
// 对图像运行3x3卷积内核,如果矩阵中的值之和不等于1.0,则图像可能变亮或变暗.
AOCE_VE_EXPORT ITLayer<Mat3x3>* create3x3ConvolutionLayer();
// Sobel边缘检测,参数高会导致边缘更牢固,默认值为1.0.
AOCE_VE_EXPORT ITLayer<float>* createSobelEdgeDetectionLayer();
// Prewitt边缘检测,参数高会导致边缘更牢固,默认值为1.0.
AOCE_VE_EXPORT ITLayer<float>* createPrewittEdgeDetectionLayer();
// 执行Sobel边缘阈值检测,根据threshold返回0/1
AOCE_VE_EXPORT ITLayer<ThresholdSobelParamet>*
createThresholdEdgeDetectionLayer();
// 执行Canny边缘阈值检测
AOCE_VE_EXPORT ITLayer<CannyEdgeDetectionParamet>*
createCannyEdgeDetectionLayer();
// Harris角点检测
AOCE_VE_EXPORT ITLayer<HarrisCornerDetectionParamet>*
createHarrisCornerDetectionLayer();
// Harris角点检测的Noble变体
AOCE_VE_EXPORT ITLayer<NobleCornerDetectionParamet>*
createNobleCornerDetectionLayer();
// Harris角点检测的Shi-Tomasi变体
AOCE_VE_EXPORT ITLayer<NobleCornerDetectionParamet>*
createShiTomasiFeatureDetectionLayer();
// 使用ColourFAST生成特征描述
AOCE_VE_EXPORT ITLayer<FASTFeatureParamet>* createColourFASTFeatureDetector();
// 用于识别标识物体十字线
// GPUImageCrosshairGenerator 现没实现

// 膨胀图像,扩大亮区域,bSingle表明是R8还是RGBA8
AOCE_VE_EXPORT ITLayer<int32_t>* createDilationLayer(bool bSingle);
// 腐蚀图像,扩大暗区域,bSingle表明是R8还是RGBA8
AOCE_VE_EXPORT ITLayer<int32_t>* createErosionLayer(bool bSingle);
// 先膨胀后腐蚀,bSingle表明是R8还是RGBA8
AOCE_VE_EXPORT ITLayer<int32_t>* createClosingLayer(bool bSingle);
// 先腐蚀后膨胀,bSingle表明是R8还是RGBA8
AOCE_VE_EXPORT ITLayer<int32_t>* createOpeningLayer(bool bSingle);
// 这将对周围8个像素的红色通道和中央像素的红色通道的强度进行比较.
// 对比较结果进行编码,得到的比特串将变为该像素强度.
// 最低有效位是右上比较,逆时针结束在最高有效位的右边比较.
AOCE_VE_EXPORT BaseLayer* createColorLBPLayer();
// 这会将低通滤波器应用于传入的视频帧.
// 基本上,这会累加前一帧和当前帧的加权滚动平均值.
// 它们可用于对视频进行降噪,添加运动模糊或用于创建高通滤波器.
// 其参数用于控制将先前累积的帧与当前帧混合的强度,范围0.0-1.0,默认0.5
AOCE_VE_EXPORT ITLayer<float>* createLowPassLayer();
// 这会将高通滤镜应用于传入的视频帧.这对于运动检测最有用.
// 这是低通滤波器的反函数,显示了当前帧与先前帧的加权滚动平均值之间的差异.
// 此选项控制将先前累积的帧进行混合然后从当前帧中减去的程度,范围0.0-1.0,默认0.5
AOCE_VE_EXPORT ITLayer<float>* createHighPassLayer();
// 这是基于高通滤波器的运动检测器.输出运动强度.
AOCE_VE_EXPORT MotionDetectorLayer* createMotionDetectorLayer();
// 使用Hough变换到平行坐标空间来检测图像中的线.
// GPUImageHoughTransformLineDetector 以后实现

// 一个帮助程序类,该类生成可以覆盖场景的线条
// GPUImageLineGenerator 以后实现

// 对图像应用定向运动模糊
AOCE_VE_EXPORT ITLayer<MotionBlurParamet>* createMotionBlurLayer();
// 将定向运动模糊应用于图像
AOCE_VE_EXPORT ITLayer<ZoomBlurParamet>* createZoomBlurLayer();
// 导向滤波
AOCE_VE_EXPORT ITLayer<GuidedParamet>* createGuidedLayer();
// 使用Laplacian算子锐化图像,bsamll参数对应不同Laplacian算子
AOCE_VE_EXPORT BaseLayer* createLaplacianLayer(bool bsamll);
#pragma endregion

// 混合模式
#pragma region Blending modes
// 有选择地将第一张图片中的颜色替换为第二张图片
// GPUImageChromaKeyBlendFilter,现在实现的VkChromaKeyBlendLayer,有考虑背景图混合,后面实现

// 应用两个图像的溶解混合,参数表示第二张图片覆盖第一张图片的程度(0.0-1.0,默认值为0.5)
AOCE_VE_EXPORT ITLayer<float>* createDissolveBlendLayer();
// 应用两个图像的多次混合
AOCE_VE_EXPORT BaseLayer* createMultiplyBlendLayer();
// 应用两个图像的加法混合
AOCE_VE_EXPORT BaseLayer* createAddBlendLayer();
// 应用两个图像的减法混合
AOCE_VE_EXPORT BaseLayer* createSubtractBlendLayer();
// 应用两个图像的除法混合
AOCE_VE_EXPORT BaseLayer* createDivideBlendLayer();
// 应用两个图像的叠加混合
AOCE_VE_EXPORT BaseLayer* createOverlayBlendLayer();
// 通过获取图像之间每个颜色分量的最小值来融合两个图像
AOCE_VE_EXPORT BaseLayer* createDarkenBlendLayer();
// 通过获取图像之间每个颜色分量的最大值来融合两个图像
AOCE_VE_EXPORT BaseLayer* createLightenBlendLayer();
// 应用两个图像的颜色加深混合
AOCE_VE_EXPORT BaseLayer* createColorBurnBlendLayer();
// 应用两个图像的颜色减淡混合
AOCE_VE_EXPORT BaseLayer* createColorDodgeBlendLayer();
// 应用两个图像的屏幕混合
AOCE_VE_EXPORT BaseLayer* createScreenBlendLayer();
// 应用两个图像的排除混合
AOCE_VE_EXPORT BaseLayer* createExclusionBlendLayer();
// 应用两个图像的差异混合
AOCE_VE_EXPORT BaseLayer* createDifferenceBlendLayer();
// 应用两个图像的强光混合
AOCE_VE_EXPORT BaseLayer* createHardLightBlendLayer();
// 应用两个图像的柔和光混合
AOCE_VE_EXPORT BaseLayer* createSoftLightBlendLayer();
// 根据第二个图像的Alpha通道在第二个图像上混合第二个图像
// 参数mix:第二张图片覆盖第一张图片的程度(0.0-1.0,默认为1.0)
AOCE_VE_EXPORT ITLayer<float>* createAlphaBlendLayer();
// 将源应用于两个图像的混合
AOCE_VE_EXPORT BaseLayer* createSourceOverBlendLayer();
// 应用两个图像的普通混合
AOCE_VE_EXPORT BaseLayer* createNormalBlendLayer();
// 应用两个图像的颜色混合
AOCE_VE_EXPORT BaseLayer* createColorBlendLayer();
// 应用两个图像的色调混合
AOCE_VE_EXPORT BaseLayer* createHueBlendLayer();
// 应用两个图像的饱和度混合
AOCE_VE_EXPORT BaseLayer* createSaturationBlendLayer();
// 应用两个图像的亮度混合
AOCE_VE_EXPORT BaseLayer* createLuminosityBlendLayer();
// 应用两个图像的线性刻录混合
AOCE_VE_EXPORT BaseLayer* createLinearBurnBlendLayer();
// 应用两个图像的泊松混合
AOCE_VE_EXPORT ITLayer<PoissonParamet>* createPoissonBlendLayer();
// 使用另一个图像遮罩一个图像
AOCE_VE_EXPORT BaseLayer* createMaskLayer();
#pragma endregion

// 视觉效果
#pragma region Visual effects
// 对图像或视频应用像素化效果
AOCE_VE_EXPORT ITLayer<PixellateParamet>* createPixellateLayer();
// 基于极坐标而不是笛卡尔坐标,对图像或视频应用像素化效果
AOCE_VE_EXPORT ITLayer<PolarPixellateParamet>* createPolarPixellateLayer();
// 选择圆形区域进行像素化效果
AOCE_VE_EXPORT ITLayer<SelectiveParamet>* createPixellatePositionLayer();
// 将图像分成规则网格内的彩色点
AOCE_VE_EXPORT ITLayer<PolkaDotParamet>* createPolkaDotLayer();
// 对图像应用半色调效果,例如新闻打印
AOCE_VE_EXPORT ITLayer<PixellateParamet>* createHalftoneLayer();
// 将图像转换为黑白交叉阴影图案
AOCE_VE_EXPORT ITLayer<CrosshatchParamet>* createCrosshatchLayer();
// 将视频转换为草图.这只是颜色反转的Sobel边缘检测滤镜
// 参数同SobelEdgeDetectionLayer
AOCE_VE_EXPORT ITLayer<float>* createSketchLayer();
// 与SketchLayer过滤器相同,仅对边缘进行阈值处理,而不是灰度
AOCE_VE_EXPORT ITLayer<ThresholdSobelParamet>* createThresholdSketchLayer();
// 使用Sobel边缘检测在对象周围放置黑色边框,然后对图像中存在的颜色进行量化,以使图像具有卡通般的质量.
AOCE_VE_EXPORT ITLayer<ToonParamet>* createToonLayer();
// 这使用与ToonLayer相似的过程,只是它在卡通效果之前带有高斯模糊用来平滑噪声
AOCE_VE_EXPORT ITLayer<SmoothToonParamet>* createSmoothToonLayer();
// 在图像上施加压纹效果,参数压花强度,从0.0到4.0,以1.0为正常水平
AOCE_VE_EXPORT ITLayer<float>* createEmbossLayer();
// 这将颜色动态范围减小到指定的步骤数,从而导致图像像卡通一样简单的阴影.
// 参数用来指定图像空间的颜色级别数.范围是1到256,默认值是10.
AOCE_VE_EXPORT ITLayer<uint32_t>* createPosterizeLayer();
// 在图像上创建漩涡形失真
AOCE_VE_EXPORT ITLayer<SwirlParamet>* createSwirlLayer();
// 在图像上创建凸出的失真
AOCE_VE_EXPORT ITLayer<DistortionParamet>* createBulgeDistortionLayer();
// 创建图像的少量变形
AOCE_VE_EXPORT ITLayer<DistortionParamet>* createPinchDistortionLayer();
// 创建图像的拉伸变形
// 参数图像的中心(在0-1.0的标准化坐标中),默认值为(0.5,0.5)
AOCE_VE_EXPORT ITLayer<vec2>* createStretchDistortionLayer();
// 模拟通过玻璃球体的折射
AOCE_VE_EXPORT ITLayer<SphereRefractionParamet>* createSphereRefractionLayer();
// 与SphereRefraction类似,仅图像未反转,玻璃的边缘有少许磨砂
AOCE_VE_EXPORT ITLayer<SphereRefractionParamet>* createGlassSphereLayer();
// 执行渐晕效果,使边缘的图像淡化
AOCE_VE_EXPORT ITLayer<VignetteParamet>* createVignetteLayer();
// 应用Kuwahara滤波,产生类似油画的图像
// 参数表示核的大小,相对GPUImage有优化,核长为3在手机Radmi K10 Pro可实时
AOCE_VE_EXPORT ITLayer<uint32_t>* createKuwaharaLayer();
// 生成充满Perlin噪点的图像
AOCE_VE_EXPORT PerlinNoiseLayer* createPerlinNoiseLayer();
// 模拟CGA监视器的颜色空间
AOCE_VE_EXPORT BaseLayer* createCGAColorspaceLayer();
// 此滤镜采用输入图块集,图块的亮度必须上升.
// GPUImageMosaicFilter 后续实现

// 生成一个Voronoi贴图,供以后使用.
// GPUImageJFAVoronoiFilter 后面实现

//接收Voronoi映射,并使用该映射过滤传入的图像.
AOCE_VE_EXPORT BaseLayer* createVoronoiConsumerLayer();
#pragma endregion

AOCE_VE_EXPORT ITLayer<ReSizeParamet>* createResizeLayer(
    ImageType imageType = ImageType::rgba8);
// 显示R8
AOCE_VE_EXPORT BaseLayer* createAlphaShowLayer();
// 显示二种输入,第一个输入R8,第二输入RGBA8的显示
AOCE_VE_EXPORT BaseLayer* createAlphaShow2Layer();
// 用于图像格式转化,包含RGBA8->RGBA32F
AOCE_VE_EXPORT BaseLayer* createConvertImageLayer();

}  // namespace vulkan
}  // namespace aoce
