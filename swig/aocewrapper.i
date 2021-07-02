// 开启代理类,使之继承回调类
%module(directors = "1") AoceWrapper

%feature("director") ILogObserver;
%feature("director") IOutputLayerObserver;
%feature("director") IMediaPlayerObserver;
%feature("director") ILiveObserver;
%feature("director") IVideoDeviceObserver;
%feature("director") ICaptureObserver; 
%feature("director") IAudioDeviceObserver;  
%{
#if __ANDROID__
#include <jni.h> 
#endif

#include "aoce/Aoce.h"
#include "aoce/AoceDefine.h"  
#include "aoce/AoceCore.h"
#include "aoce_vulkan_extra/AoceVkExtra.h"
#include "aoce_vulkan_extra/VkExtraExport.h"

%}

#define ACOE_EXPORT
#define AOCE_VE_EXPORT

%ignore aoce::logEventAction;
%ignore aoce::setLogAction;
%ignore aoce::IVideoManager::getDevices(IVideoDevice**, int32_t);
%ignore aoce::IVideoManager::getDevices(IVideoDevice**, int32_t,int32_t);
%ignore aoce::IVideoDevice::getFormats;
%ignore aoce::VkOutGpuTex;
%ignore aoce::IOutputLayer::outVkGpuTex(const VkOutGpuTex& ,int32_t);
%ignore aoce::IOutputLayer::outVkGpuTex(const VkOutGpuTex&);
%ignore operator==; 
%ignore operator[];
%ignore operator*;

#ifdef SWIGJAVA
// https://forge.naos-cluster.tech/aquinetic/f2i-consulting/fesapi/-/blob/c0a52292680e4ec316d2e3447b52f365a54cc400/cmake/swigModule.i
// getCPtr由protected改成public
SWIG_JAVABODY_PROXY(protected, public, SWIGTYPE)
SWIG_JAVABODY_TYPEWRAPPER(public, public, public, SWIGTYPE)
#endif

#ifdef SWIGCSHARP
#endif

// 将C++ 中 void*/uint8_t*转C# IntPtr
%apply void *VOID_INT_PTR { void *,uint8_t * }
// 没有的话,int32_t对应不了int
%include "stdint.i"
// %include "arrays_csharp.i"
%include "carrays.i" 
%include "windows.i"
// IntArray<->SWIGTYPE_p_int
%array_class(int, IntArray);
%array_class(uint8_t, UInt8Array);
//如下文件生成包装
%include "aoce/Aoce.h"
%include "aoce/AoceLive.h"
%include "aoce/AoceMath.h"
%include "aoce/AoceMedia.h"
%include "aoce/AoceVideoDevice.h"
%include "aoce/AoceLayer.h"
%include "aoce/AoceMetadata.h"  
%include "aoce/AoceWindow.h"  
%include "aoce/AoceAudioDevice.h"  
// 针对ITLayer需预先实例化二个类告诉swig
%template(AInputLayer) aoce::ITLayer<aoce::InputParamet>;
%template(AOutputLayer) aoce::ITLayer<aoce::OutputParamet>; 
%include "aoce/AoceCore.h" 
%template(IYUVLayer) aoce::ITLayer<aoce::YUVParamet>; 
%template(IMapChannelLayer) aoce::ITLayer<aoce::MapChannelParamet>;
%template(IFlipLayer) aoce::ITLayer<aoce::FlipParamet>;
%template(ITransposeLayer) aoce::ITLayer<aoce::TransposeParamet>;
%template(IReSizeLayer) aoce::ITLayer<aoce::ReSizeParamet>;
%template(IBlendLayer) aoce::ITLayer<aoce::BlendParamet>;

%template(ILBoolMetadata) aoce::ILTMetadata<bool>; 
%template(ILStringMetadata) aoce::ILTMetadata<const char*>; 
%template(ILIntMetadata) aoce::ILTRangeMetadata<int32_t>; 
%template(ILFloatMetadata) aoce::ILTRangeMetadata<float>; 

// %extend aoce::BGroupMetadata {
//     static aoce::BGroupMetadata* dynamic_cast(aoce::ILMetadata *lmeta) {
//         return dynamic_cast<aoce::BGroupMetadata*>(lmeta);
//     }
// };

%template(ASoftEleganceLayer) aoce::ITLayer<aoce::SoftEleganceParamet>;
%template(AFloatLayer) aoce::ITLayer<float>;
%template(APerlinNoiseLayer) aoce::ITLayer<aoce::PerlinNoiseParamet>;
%include "aoce_vulkan_extra/AoceVkExtra.h"
%include "aoce_vulkan_extra/VkExtraExport.h"
%template(IStretchDistortionLayer) aoce::ITLayer<aoce::vec2> ;
%template(IRGBLayer) aoce::ITLayer<aoce::vec3> ;
%template(IMedianLayer) aoce::ITLayer<uint32_t> ;
%template(I3x3ConvolutionLayer) aoce::ITLayer<aoce::Mat3x3> ;
%template(IMorphLayer) aoce::ITLayer<int32_t> ;
%template(ISizeScaleLayer) aoce::ITLayer<aoce::SizeScaleParamet> ;
%template(IKernelSizeLayer) aoce::ITLayer<aoce::KernelSizeParamet> ;
%template(IGaussianBlurLayer) aoce::ITLayer<aoce::GaussianBlurParamet> ;
%template(IChromaKeyLayer) aoce::ITLayer<aoce::ChromaKeyParamet> ;
%template(IAdaptiveThresholdLayer) aoce::ITLayer<aoce::AdaptiveThresholdParamet> ;
%template(IGuidedLayer) aoce::ITLayer<aoce::GuidedParamet> ;
%template(IHarrisCornerDetectionLayer) aoce::ITLayer<aoce::HarrisCornerDetectionParamet> ;
%template(INobleCornerDetectionLayer) aoce::ITLayer<aoce::NobleCornerDetectionParamet> ;
%template(ICannyEdgeDetectionLayer) aoce::ITLayer<aoce::CannyEdgeDetectionParamet> ;
%template(IFASTFeatureLayer) aoce::ITLayer<aoce::FASTFeatureParamet> ;
%template(IBilateralLayer) aoce::ITLayer<aoce::BilateralParamet> ;
%template(IDistortionLayer) aoce::ITLayer<aoce::DistortionParamet> ;
%template(IPositionLayer) aoce::ITLayer<aoce::PositionParamet> ;
%template(ISelectiveLayer) aoce::ITLayer<aoce::SelectiveParamet> ;
%template(IBulrPositionLayer) aoce::ITLayer<aoce::BulrPositionParamet> ;
%template(IBlurSelectiveLayer) aoce::ITLayer<aoce::BlurSelectiveParamet> ;
%template(ISphereRefractionLayer) aoce::ITLayer<aoce::SphereRefractionParamet> ;
%template(IPixellateLayer) aoce::ITLayer<aoce::PixellateParamet> ;
%template(IColorMatrixLayer) aoce::ITLayer<aoce::ColorMatrixParamet> ;
%template(ICropLayer) aoce::ITLayer<aoce::CropParamet> ;
%template(ICrosshatchLayer) aoce::ITLayer<aoce::CrosshatchParamet> ;
%template(IFalseColorLayer) aoce::ITLayer<aoce::FalseColorParamet> ;
%template(IHazeLayer) aoce::ITLayer<aoce::HazeParamet> ;
%template(IHighlightShadowLayer) aoce::ITLayer<aoce::HighlightShadowParamet> ;
%template(IHighlightShadowTintLayer) aoce::ITLayer<aoce::HighlightShadowTintParamet> ;
%template(IIOSBlurLayer) aoce::ITLayer<aoce::IOSBlurParamet> ;
%template(ILevelsLayer) aoce::ITLayer<aoce::LevelsParamet> ;
%template(IMonochromeLayer) aoce::ITLayer<aoce::MonochromeParamet> ;
%template(IMotionBlurLayer) aoce::ITLayer<aoce::MotionBlurParamet> ;
%template(IPoissonLayer) aoce::ITLayer<aoce::PoissonParamet> ;
%template(IPolarPixellateLayer) aoce::ITLayer<aoce::PolarPixellateParamet> ;
%template(IPolkaDotLayer) aoce::ITLayer<aoce::PolkaDotParamet> ;
%template(ISharpenLayer) aoce::ITLayer<aoce::SharpenParamet> ;
%template(ISkinToneLayer) aoce::ITLayer<aoce::SkinToneParamet> ;
%template(IToonLayer) aoce::ITLayer<aoce::ToonParamet> ;
%template(ISmoothToonLayer) aoce::ITLayer<aoce::SmoothToonParamet> ;
%template(ISwirlLayer) aoce::ITLayer<aoce::SwirlParamet> ;
%template(IThresholdSobelLayer) aoce::ITLayer<aoce::ThresholdSobelParamet> ;
%template(ITiltShiftLayer) aoce::ITLayer<aoce::TiltShiftParamet> ;
%template(IUnsharpMaskLayer) aoce::ITLayer<aoce::UnsharpMaskParamet> ;
%template(IVignetteLayer) aoce::ITLayer<aoce::VignetteParamet> ;
%template(IWhiteBalanceLayer) aoce::ITLayer<aoce::WhiteBalanceParamet> ;
%template(IZoomBlurLayer) aoce::ITLayer<aoce::ZoomBlurParamet> ;


%nodefaultctor;
%nodefaultdtor;

%clearnodefaultctor;
%clearnodefaultdtor;






