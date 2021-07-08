package aoce.samples.aoceswigtest;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.Message;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;


import aoce.android.library.JNIHelper;
import aoce.android.library.xswig.*;
import aoce.samples.aoceswigtest.aocewrapper.AoceManager;

public class DataManager {
    private AoceManager aoceManager = null;
    // Color adjustments
    private AFloatLayer brightLayer = null;
    private AFloatLayer exposureLayer = null;
    private AFloatLayer contrastLayer = null;
    private AFloatLayer saturationLayer = null;
    private AFloatLayer gammaLayer = null;
    private AFloatLayer solarizeLayer = null;
    private ILevelsLayer levelsLayer = null;
    private IColorMatrixLayer colorMatrixLayer = null;
    private IRGBLayer rgbLayer = null;
    private AFloatLayer hueLayer = null;
    private AFloatLayer vibranceLayer = null;
    private IWhiteBalanceLayer whiteBalanceLayer = null;
    private IHighlightShadowLayer highlightShadowLayer = null;
    private IHighlightShadowTintLayer highlightShadowTintLayer = null;
    private ILookupLayer lookupLayer = null;
    private ISoftEleganceLayer softEleganceLayer = null;
    private ISkinToneLayer skinToneLayer = null;
    private IBaseLayer colorInverLayer = null;
    private IBaseLayer luminanceLayer = null;
    private IMonochromeLayer monochromeLayer = null;
    private IFalseColorLayer falseColorLayer = null;
    private IHazeLayer hazeLayer = null;
    private AFloatLayer sepiaLayer = null;
    private AFloatLayer opacityLayer = null;
    private AFloatLayer luminanceThresholdLayer = null;
    private IAdaptiveThresholdLayer adaptiveThresholdLayer = null;
    private AFloatLayer averageLuminanceThresholdLayer = null;
    private IBaseLayer singleHistoramlayer = null;
    private IBaseLayer historamlayer = null;
    private IChromaKeyLayer chromaKeyLayer = null;
    private IHSBLayer hsbLayer = null;
    // Image processing
    private ISharpenLayer sharpenLayer = null;
    private IUnsharpMaskLayer unsharpMaskLayer = null;
    private IGaussianBlurLayer gaussianBlurLayer = null;
    private IKernelSizeLayer boxBlurLayer = null;
    private IBlurSelectiveLayer blurSelectiveLayer = null;
    private IBlurPositionLayer blurPositionLayer = null;
    private IIOSBlurLayer iosBlurLayer = null;
    private IBaseLayer medialLayer = null;
    private IBilateralLayer bilateralLayer = null;
    private ITiltShiftLayer tiltShiftLayer = null;
    private AFloatLayer sobelEdgeDetectionLayer = null;
    private AFloatLayer prewittEdgeDetectionLayer = null;
    private IThresholdSobelLayer thresholdEdgeDetectionLayer = null;
    private ICannyEdgeDetectionLayer cannyEdgeDetectionLayer = null;
    private IHarrisCornerDetectionLayer harrisCornerDetectionLayer = null;
    private INobleCornerDetectionLayer nobleCornerDetectionLayer = null;
    private INobleCornerDetectionLayer shiTomasiDetectionLayer = null;
    private IFASTFeatureLayer fastFeatureLayer = null;
    private IMorphLayer dilationLayer = null;
    private IMorphLayer erosionLayer = null;
    private IMorphLayer closingLayer = null;
    private IMorphLayer openingLayer = null;
    private IMorphLayer singleDilationLayer = null;
    private IMorphLayer singleErosionLayer = null;
    private IMorphLayer singleClosingLayer = null;
    private IMorphLayer singleOpeningLayer = null;
    private IBaseLayer colorLBPLayer = null;
    private AFloatLayer lowPassLayer = null;
    private AFloatLayer highPassLayer = null;
    private IMotionDetectorLayer motionDetectorLayer = null;
    private MotionDetectorObserver motionDetectorObserver = new MotionDetectorObserver();
    private IMotionBlurLayer motionBlurLayer = null;
    private IZoomBlurLayer zoomBlurLayer = null;
    private IGuidedLayer guidedLayer = null;
    private IBaseLayer laplacianLayer = null;
    // Blending modes
    private AFloatLayer dissolveBlendLayer = null;
    private IBaseLayer multiplyBlendLayer = null;
    private IBaseLayer addBlendLayer = null;
    private IBaseLayer subtractBlendLayer = null;
    private IBaseLayer divideBlendLayer = null;
    private IBaseLayer overlayBlendLayer = null;
    private IBaseLayer darkenBlendLayer = null;
    private IBaseLayer lightenBlendLayer = null;
    private IBaseLayer colorBurnBlendLayer = null;
    private IBaseLayer colorDodgeBlendLayer = null;
    private IBaseLayer screenBlendLayer = null;
    private IBaseLayer exclusionBlendLayer = null;
    private IBaseLayer differenceBlendLayer = null;
    private IBaseLayer hardLightBlendLayer = null;
    private IBaseLayer softLightBlendLayer = null;
    private AFloatLayer alphaBlendLayer = null;
    private IBaseLayer sourceOverBlendLayer = null;
    private IBaseLayer normalBlendLayer = null;
    private IBaseLayer colorBlendLayer = null;
    private IBaseLayer hueBlendLayer = null;
    private IBaseLayer saturationBlendLayer = null;
    private IBaseLayer luminosityBlendLayer = null;
    private IBaseLayer linearBurnBlendLayer = null;
    private IPoissonLayer poissonLayer = null;
    private IBaseLayer maskLayer = null;
    // Visual effects
    private IPixellateLayer pixellateLayer = null;
    private IPolarPixellateLayer polarPixellateLayer = null;
    private ISelectiveLayer pixellatePositionLayer = null;
    private IPolkaDotLayer polkaDotLayer = null;
    private IPixellateLayer halftoneLayer = null;
    private ICrosshatchLayer crosshatchLayer = null;
    private AFloatLayer sketchLayer = null;
    private IThresholdSobelLayer thresholdSketchLayer = null;
    private IToonLayer toonLayer = null;
    private ISmoothToonLayer smoothToonLayer = null;
    private AFloatLayer embossLayer = null;
    private IMedianLayer posterizeLayer = null;
    private ISwirlLayer swirlLayer = null;
    private IDistortionLayer bulgeDistortionLayer = null;
    private IDistortionLayer pinchDistortionLayer = null;
    private IStretchDistortionLayer stretchDistortionLayer = null;
    private ISphereRefractionLayer sphereRefractionLayer = null;
    private ISphereRefractionLayer glassSphereLayer = null;
    private IVignetteLayer vignetteLayer = null;
    private IMedianLayer kuwaharaLayer = null;
    private IPerlinNoiseLayer perlinNoiseLayer = null;
    private IBaseLayer cgaColorspaceLayer = null;
    // common layer
    private IBaseLayer alphaShowLayer = null;
    private IBaseLayer alphaShowLayer2 = null;
    private IKernelSizeLayer edgeBoxLayer = null;
    private IInputLayer inputBlendLayer = null;
    private Handler handler = null;

    public static final int MONION_UPDATE = 615;
    private String blendImagePath = "blend.png";

    public enum LayerType {
        normal,
        edgeDetection,
        twoInput,
        blend,
    }

    public class LayerItem {
        public String name = "";
        public String metadata = "";
        public int layerIndex = 0;
        public ArrayList<Object> layers = new ArrayList<>();
        private Object paramet = null;
        public LayerType layerType = LayerType.normal;

        public ArrayList<IBaseLayer> getLayers() {
            ArrayList<IBaseLayer> baseLayers = new ArrayList<>();
            for (Object layer : layers) {
                if (layer instanceof IBaseLayer) {
                    baseLayers.add((IBaseLayer) layer);
                } else if (layer instanceof ILayer) {
                    baseLayers.add(((ILayer) layer).getLayer());
                }
            }
            return baseLayers;
        }

        public Object getLayer() {
            return layers.get(layerIndex);
        }

        public Object getParamet() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
            Object layer = getLayer();
            Method getMethod = layer.getClass().getMethod("getParamet");
            paramet = getMethod.invoke(layer);
            return paramet;
        }

        public void updateParamet() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
            Object layer = getLayer();
            Method setMethod = layer.getClass().getMethod("updateParamet", getPrimitive(paramet.getClass()));
            setMethod.invoke(layer, paramet);
        }
    }

    public class LayerGroup {
        public String title = "";
        public ArrayList<LayerItem> layers = new ArrayList<LayerItem>();

        public LayerItem addItem(String itemName, String metadataname, Object... blayers) {
            LayerItem item = new LayerItem();
            item.name = itemName;
            item.metadata = metadataname;
            for (int i = 0; i < blayers.length; i++) {
                item.layers.add(blayers[i]);
            }
            layers.add(item);
            return item;
        }

        public void addEdgeItem(String itemName, String metadataname, Object layer) {
            LayerItem item = addItem(itemName, metadataname, layer);
            item.layerType = LayerType.edgeDetection;
        }

        public void addTwoItem(String itemName, String metadataname, Object layer) {
            LayerItem item = addItem(itemName, metadataname, layer);
            item.layerType = LayerType.twoInput;
        }

        public void addItemLum(String itemName, String metadataname, Object... blayers) {
            LayerItem item = addItem(itemName, metadataname, blayers);
            item.layerIndex = 1;
        }

        public void addBlendItem(String itemName, String metadataname, Object layer) {
            LayerItem item = addItem(itemName, metadataname, layer);
            item.layerType = LayerType.blend;
        }
    }

    public class MotionDetectorObserver extends IMotionDetectorObserver {
        @Override
        public void onMotion(vec4 vec) {
            if (handler != null) {
                Message message = new Message();
                message.what = MONION_UPDATE;
                message.arg1 = (int) vec.getX();
                message.arg2 = (int) vec.getY();
                handler.sendMessage(message);
            }
        }
    }

    private ArrayList<LayerGroup> groups = new ArrayList<LayerGroup>();

    private static class DataManagerInstance {
        private static final DataManager instance = new DataManager();
    }

    private DataManager() {
        AoceWrapper.initPlatform();
        AoceWrapper.loadAoce();
        aoceManager = new AoceManager();
        aoceManager.initGraph();

        alphaShowLayer = AoceWrapper.createAlphaShowLayer();
        inputBlendLayer = AoceWrapper.getLayerFactory(GpuType.vulkan).createInput();
        alphaShowLayer2 = AoceWrapper.createAlphaShow2Layer();
        edgeBoxLayer = AoceWrapper.createBoxFilterLayer(ImageType.r8);
        KernelSizeParamet kernelSizeParamet = edgeBoxLayer.getParamet();
        kernelSizeParamet.setKernelSizeX(5);
        kernelSizeParamet.setKernelSizeY(5);
        edgeBoxLayer.updateParamet(kernelSizeParamet);

        initColorAdjustmentsLayer();
        initImageProcessingLayer();
        initBlendingModesLayer();
        initVisualEffectsLayer();
    }

    private void initColorAdjustmentsLayer() {
        brightLayer = AoceWrapper.createBrightnessLayer();
        exposureLayer = AoceWrapper.createExposureLayer();
        contrastLayer = AoceWrapper.createContrastLayer();
        saturationLayer = AoceWrapper.createSaturationLayer();
        gammaLayer = AoceWrapper.createGammaLayer();
        solarizeLayer = AoceWrapper.createSolarizeLayer();
        levelsLayer = AoceWrapper.createLevelsLayer();
        rgbLayer = AoceWrapper.createRGBLayer();
        hueLayer = AoceWrapper.createHueLayer();
        vibranceLayer = AoceWrapper.createVibranceLayer();
        whiteBalanceLayer = AoceWrapper.createBalanceLayer();
        highlightShadowLayer = AoceWrapper.createHighlightShadowLayer();
        highlightShadowTintLayer = AoceWrapper.createHighlightShadowTintLayer();
        lookupLayer = AoceWrapper.createLookupLayer();
        softEleganceLayer = AoceWrapper.createSoftEleganceLayer();
        skinToneLayer = AoceWrapper.createSkinToneLayer();
        colorInverLayer = AoceWrapper.createColorInvertLayer();
        luminanceLayer = AoceWrapper.createLuminanceLayer();
        monochromeLayer = AoceWrapper.createMonochromeLayer();
        falseColorLayer = AoceWrapper.createFalseColorLayer();
        hazeLayer = AoceWrapper.createHazeLayer();
        sepiaLayer = AoceWrapper.createSepiaLayer();
        opacityLayer = AoceWrapper.createOpacityLayer();
        luminanceThresholdLayer = AoceWrapper.createLuminanceThresholdLayer();
        adaptiveThresholdLayer = AoceWrapper.createAdaptiveThresholdLayer();
        averageLuminanceThresholdLayer = AoceWrapper.createAverageLuminanceThresholdLayer();
        singleHistoramlayer = AoceWrapper.createHistogramLayer(true);
        historamlayer = AoceWrapper.createHistogramLayer(false);
        chromaKeyLayer = AoceWrapper.createChromaKeyLayer();
        hsbLayer = AoceWrapper.createHSBLayer();

        LayerGroup layerGroup = new LayerGroup();
        layerGroup.title = "色彩调整";
        layerGroup.addItem("亮度", "brightnessLayer", brightLayer);
        layerGroup.addItem("曝光度", "exposureLayer", exposureLayer);
        layerGroup.addItem("对比度", "contrastLayer", contrastLayer);
        layerGroup.addItem("饱和度", "saturationLayer", saturationLayer);
        layerGroup.addItem("Gamma", "gammaLayer", gammaLayer);
        layerGroup.addItem("基于阈值反转颜色", "solarizeLayer", solarizeLayer);
        layerGroup.addItem("色阶调整", "levelsLayer", levelsLayer);
        layerGroup.addItem("调整RGB分量", "rgbLayer", rgbLayer);
        layerGroup.addItem("色调", "hueLayer", hueLayer);
        layerGroup.addItem("鲜艳度", "vibranceLayer", vibranceLayer);
        layerGroup.addItem("白平衡", "whiteBalanceLayer", whiteBalanceLayer);
        layerGroup.addItem("阴影与高光", "highlightShadowLayer", highlightShadowLayer);
        layerGroup.addItem("基于颜色和强度的阴影与高光", "highlightShadowTintLayer", highlightShadowTintLayer);
        layerGroup.addItem("Amatorka滤镜(Lookup)", "lookupLayer", lookupLayer);
        layerGroup.addItem("MissEtikate滤镜(Lookup)", "lookupLayer", lookupLayer);
        layerGroup.addItem("SoftElegance滤镜(Lookup)", "softEleganceLayer", softEleganceLayer);
        layerGroup.addItem("肤色调整滤镜", "skinToneLayer", skinToneLayer);
        layerGroup.addItem("反转图像", "colorInverLayer", colorInverLayer);
        layerGroup.addItem("灰度", "luminanceLayer", luminanceLayer, alphaShowLayer);
        layerGroup.addItem("基于亮度的单色", "monochromeLayer", monochromeLayer);
        layerGroup.addItem("基于亮度的混合", "falseColorLayer", falseColorLayer);
        layerGroup.addItem("调整雾度", "hazeLayer", hazeLayer);
        layerGroup.addItem("棕褐色调", "sepiaLayer", sepiaLayer);
        layerGroup.addItem("基于亮度的阈值", "luminanceThresholdLayer", luminanceThresholdLayer, alphaShowLayer);
        layerGroup.addItem("基于周边平均亮度自适应阈值", "adaptiveThresholdLayer", adaptiveThresholdLayer, alphaShowLayer);
        layerGroup.addItem("基于全图平均亮度的阈值", "averageLuminanceThresholdLayer", averageLuminanceThresholdLayer, alphaShowLayer);
        layerGroup.addItem("直方图", "historamlayer", historamlayer, alphaShowLayer);
        layerGroup.addItem("直方图(单通道)", "historamlayer", luminanceLayer, singleHistoramlayer, alphaShowLayer);
        layerGroup.addItem("色度扣像", "chromaKeyLayer", chromaKeyLayer, alphaShowLayer);
        layerGroup.addItem("色相/饱和度/亮度", "hsbLayer", hsbLayer);
        groups.add(layerGroup);
    }

    private void initImageProcessingLayer() {
        sharpenLayer = AoceWrapper.createSharpenLayer();
        unsharpMaskLayer = AoceWrapper.createUnsharpMaskLayer();
        gaussianBlurLayer = AoceWrapper.createGaussianBlurLayer(ImageType.rgba8);
        boxBlurLayer = AoceWrapper.createBoxFilterLayer(ImageType.rgba8);
        blurSelectiveLayer = AoceWrapper.createBlurSelectiveLayer();
        blurPositionLayer = AoceWrapper.createBlurPositionLayer();
        iosBlurLayer = AoceWrapper.createIOSBlurLayer();
        medialLayer = AoceWrapper.createMedianK3Layer(false);
        bilateralLayer = AoceWrapper.createBilateralLayer();
        tiltShiftLayer = AoceWrapper.createTiltShiftLayer();
        sobelEdgeDetectionLayer = AoceWrapper.createSobelEdgeDetectionLayer();
        prewittEdgeDetectionLayer = AoceWrapper.createPrewittEdgeDetectionLayer();
        thresholdEdgeDetectionLayer = AoceWrapper.createThresholdEdgeDetectionLayer();
        cannyEdgeDetectionLayer = AoceWrapper.createCannyEdgeDetectionLayer();
        harrisCornerDetectionLayer = AoceWrapper.createHarrisCornerDetectionLayer();
        nobleCornerDetectionLayer = AoceWrapper.createNobleCornerDetectionLayer();
        shiTomasiDetectionLayer = AoceWrapper.createShiTomasiFeatureDetectionLayer();
        fastFeatureLayer = AoceWrapper.createColourFASTFeatureDetector();
        dilationLayer = AoceWrapper.createDilationLayer(false);
        erosionLayer = AoceWrapper.createErosionLayer(false);
        closingLayer = AoceWrapper.createClosingLayer(false);
        openingLayer = AoceWrapper.createOpeningLayer(false);
        singleDilationLayer = AoceWrapper.createDilationLayer(true);
        singleErosionLayer = AoceWrapper.createErosionLayer(true);
        singleClosingLayer = AoceWrapper.createClosingLayer(true);
        singleOpeningLayer = AoceWrapper.createOpeningLayer(true);
        colorLBPLayer = AoceWrapper.createColorLBPLayer();
        lowPassLayer = AoceWrapper.createLowPassLayer();
        highPassLayer = AoceWrapper.createHighPassLayer();
        motionDetectorLayer = AoceWrapper.createMotionDetectorLayer();
        motionDetectorLayer.setObserver(motionDetectorObserver);
        motionBlurLayer = AoceWrapper.createMotionBlurLayer();
        zoomBlurLayer = AoceWrapper.createZoomBlurLayer();
        guidedLayer = AoceWrapper.createGuidedLayer();
        laplacianLayer = AoceWrapper.createLaplacianLayer(false);

        LayerItem item = null;
        LayerGroup layerGroup = new LayerGroup();
        layerGroup.title = "图像处理";
        layerGroup.addItem("锐化图像", "sharpenLayer", sharpenLayer);
        layerGroup.addItem("模糊蒙版", "unsharpMaskLayer", unsharpMaskLayer);
        layerGroup.addItem("高斯模糊", "gaussianBlurLayer", gaussianBlurLayer);
        layerGroup.addItem("Box模糊", "boxBlurLayer", boxBlurLayer);
        layerGroup.addItem("特定圆形区域清晰", "blurSelectiveLayer", blurSelectiveLayer);
        layerGroup.addItem("特定圆形区域模糊", "blurPositionLayer", blurPositionLayer);
        layerGroup.addItem("IOS模糊", "iosBlurLayer", iosBlurLayer);
        layerGroup.addItem("中值模糊", "medialLayer", medialLayer);
        layerGroup.addItem("双边滤波", "bilateralLayer", bilateralLayer);
        layerGroup.addItem("模拟倾斜移位镜头效果", "tiltShiftLayer", tiltShiftLayer);
        layerGroup.addItemLum("Sobel边缘检测", "sobelEdgeDetectionLayer", luminanceLayer, sobelEdgeDetectionLayer, alphaShowLayer);
        layerGroup.addItemLum("Prewitt边缘检测", "prewittEdgeDetectionLayer", luminanceLayer, prewittEdgeDetectionLayer, alphaShowLayer);
        layerGroup.addItemLum("Sobel边缘阈值检测", "thresholdEdgeDetectionLayer", luminanceLayer, thresholdEdgeDetectionLayer, alphaShowLayer);
        layerGroup.addItem("Canny边缘阈值检测", "cannyEdgeDetectionLayer", cannyEdgeDetectionLayer, alphaShowLayer);
        layerGroup.addEdgeItem("Harris角点检测", "harrisCornerDetectionLayer", harrisCornerDetectionLayer);
        layerGroup.addEdgeItem("Noble角点检测", "nobleCornerDetectionLayer", nobleCornerDetectionLayer);
        layerGroup.addEdgeItem("Shi-Tomasi角点检测", "nobleCornerDetectionLayer", shiTomasiDetectionLayer);
        layerGroup.addItem("ColourFAST特征描述", "fastFeatureLayer", fastFeatureLayer);
        layerGroup.addItem("膨胀图像", "dilationLayer", dilationLayer);
        layerGroup.addItem("腐蚀图像", "erosionLayer", erosionLayer);
        layerGroup.addItem("先膨胀后腐蚀(闭运算)", "closingLayer", closingLayer);
        layerGroup.addItem("先腐蚀后膨胀(开运算)", "openingLayer", openingLayer);
        layerGroup.addItemLum("膨胀图像(单通道)", "dilationLayer", luminanceLayer, singleDilationLayer, alphaShowLayer);
        layerGroup.addItemLum("腐蚀图像(单通道)", "erosionLayer", luminanceLayer, singleErosionLayer, alphaShowLayer);
        layerGroup.addItemLum("先膨胀后腐蚀(单通道)", "closingLayer", luminanceLayer, singleClosingLayer, alphaShowLayer);
        layerGroup.addItemLum("先腐蚀后膨胀(单通道)", "openingLayer", luminanceLayer, singleOpeningLayer, alphaShowLayer);
        layerGroup.addItem("LBP像素编码", "colorLBPLayer", colorLBPLayer);
        layerGroup.addItem("低通滤波器", "lowPassLayer", lowPassLayer);
        layerGroup.addTwoItem("高通滤波器", "highPassLayer", highPassLayer);
        layerGroup.addItem("运动检测器", "motionDetectorLayer", motionDetectorLayer);
        layerGroup.addItem("定向运动模糊", "motionBlurLayer", motionBlurLayer);
        layerGroup.addItem("中心运动模糊", "zoomBlurLayer", zoomBlurLayer);
        layerGroup.addItem("扣像+导向滤波", "chromaKeyLayer", chromaKeyLayer, guidedLayer, alphaShowLayer);
        layerGroup.addItem("Laplacian锐化", "laplacianLayer", laplacianLayer);

        groups.add(layerGroup);
    }

    private void initBlendingModesLayer() {
        dissolveBlendLayer = AoceWrapper.createDissolveBlendLayer();
        multiplyBlendLayer = AoceWrapper.createMultiplyBlendLayer();
        addBlendLayer = AoceWrapper.createAddBlendLayer();
        subtractBlendLayer = AoceWrapper.createSubtractBlendLayer();
        divideBlendLayer = AoceWrapper.createDivideBlendLayer();
        overlayBlendLayer = AoceWrapper.createOverlayBlendLayer();
        darkenBlendLayer = AoceWrapper.createDarkenBlendLayer();
        lightenBlendLayer = AoceWrapper.createLightenBlendLayer();
        colorBurnBlendLayer = AoceWrapper.createColorBurnBlendLayer();
        colorDodgeBlendLayer = AoceWrapper.createColorDodgeBlendLayer();
        screenBlendLayer = AoceWrapper.createScreenBlendLayer();
        exclusionBlendLayer = AoceWrapper.createExclusionBlendLayer();
        differenceBlendLayer = AoceWrapper.createDifferenceBlendLayer();
        hardLightBlendLayer = AoceWrapper.createHardLightBlendLayer();
        softLightBlendLayer = AoceWrapper.createSoftLightBlendLayer();
        alphaBlendLayer = AoceWrapper.createAlphaBlendLayer();
        sourceOverBlendLayer = AoceWrapper.createSourceOverBlendLayer();
        normalBlendLayer = AoceWrapper.createNormalBlendLayer();
        colorBlendLayer = AoceWrapper.createColorBlendLayer();
        hueBlendLayer = AoceWrapper.createHueBlendLayer();
        saturationBlendLayer = AoceWrapper.createSaturationBlendLayer();
        luminosityBlendLayer = AoceWrapper.createLuminosityBlendLayer();
        linearBurnBlendLayer = AoceWrapper.createLinearBurnBlendLayer();
        poissonLayer = AoceWrapper.createPoissonBlendLayer();
        maskLayer = AoceWrapper.createMaskLayer();

        LayerGroup layerGroup = new LayerGroup();
        layerGroup.title = "混合模式";
        layerGroup.addBlendItem("溶解混合", "dissolveBlendLayer", dissolveBlendLayer);
        layerGroup.addBlendItem("多次混合", "multiplyBlendLayer", multiplyBlendLayer);
        layerGroup.addBlendItem("加法混合", "addBlendLayer", addBlendLayer);
        layerGroup.addBlendItem("减法混合", "subtractBlendLayer", subtractBlendLayer);
        layerGroup.addBlendItem("除法混合", "divideBlendLayer", divideBlendLayer);
        layerGroup.addBlendItem("叠加混合", "overlayBlendLayer", overlayBlendLayer);
        layerGroup.addBlendItem("分量最小混合", "darkenBlendLayer", darkenBlendLayer);
        layerGroup.addBlendItem("分量最大混合", "lightenBlendLayer", lightenBlendLayer);
        layerGroup.addBlendItem("加深混合", "colorBurnBlendLayer", colorBurnBlendLayer);
        layerGroup.addBlendItem("减淡混合", "colorDodgeBlendLayer", colorDodgeBlendLayer);
        layerGroup.addBlendItem("屏幕混合", "screenBlendLayer", screenBlendLayer);
        layerGroup.addBlendItem("排除混合", "exclusionBlendLayer", exclusionBlendLayer);
        layerGroup.addBlendItem("差异混合", "differenceBlendLayer", differenceBlendLayer);
        layerGroup.addBlendItem("强化混合", "hardLightBlendLayer", hardLightBlendLayer);
        layerGroup.addBlendItem("柔和光混合", "softLightBlendLayer", softLightBlendLayer);
        layerGroup.addBlendItem("Alpha混合", "alphaBlendLayer", alphaBlendLayer);
        layerGroup.addBlendItem("图像源混合", "sourceOverBlendLayer", sourceOverBlendLayer);
        layerGroup.addBlendItem("普通混合", "normalBlendLayer", normalBlendLayer);
        layerGroup.addBlendItem("图像混合", "colorBlendLayer", colorBlendLayer);
        layerGroup.addBlendItem("色调混合", "hueBlendLayer", hueBlendLayer);
        layerGroup.addBlendItem("饱和度混合", "saturationBlendLayer", saturationBlendLayer);
        layerGroup.addBlendItem("亮度混合", "luminosityBlendLayer", luminosityBlendLayer);
        layerGroup.addBlendItem("线性刻录混合", "linearBurnBlendLayer", linearBurnBlendLayer);
        layerGroup.addBlendItem("泊松混合", "poissonLayer", poissonLayer);
        layerGroup.addBlendItem("遮罩显示", "maskLayer", maskLayer);

        groups.add(layerGroup);
    }

    private void initVisualEffectsLayer() {
        pixellateLayer = AoceWrapper.createPixellateLayer();
        polarPixellateLayer = AoceWrapper.createPolarPixellateLayer();
        pixellatePositionLayer = AoceWrapper.createPixellatePositionLayer();
        polkaDotLayer = AoceWrapper.createPolkaDotLayer();
        halftoneLayer = AoceWrapper.createHalftoneLayer();
        crosshatchLayer = AoceWrapper.createCrosshatchLayer();
        sketchLayer = AoceWrapper.createSketchLayer();
        thresholdSketchLayer = AoceWrapper.createThresholdSketchLayer();
        toonLayer = AoceWrapper.createToonLayer();
        smoothToonLayer = AoceWrapper.createSmoothToonLayer();
        embossLayer = AoceWrapper.createEmbossLayer();
        posterizeLayer = AoceWrapper.createPosterizeLayer();
        swirlLayer = AoceWrapper.createSwirlLayer();
        bulgeDistortionLayer = AoceWrapper.createBulgeDistortionLayer();
        pinchDistortionLayer = AoceWrapper.createPinchDistortionLayer();
        stretchDistortionLayer = AoceWrapper.createStretchDistortionLayer();
        sphereRefractionLayer = AoceWrapper.createSphereRefractionLayer();
        glassSphereLayer = AoceWrapper.createGlassSphereLayer();
        vignetteLayer = AoceWrapper.createVignetteLayer();
        kuwaharaLayer = AoceWrapper.createKuwaharaLayer();
        kuwaharaLayer.updateParamet(3);
        perlinNoiseLayer = AoceWrapper.createPerlinNoiseLayer();
        cgaColorspaceLayer = AoceWrapper.createCGAColorspaceLayer();

        LayerGroup layerGroup = new LayerGroup();
        layerGroup.title = "视觉效果";
        layerGroup.addItem("像素化", "pixellateLayer", pixellateLayer);
        layerGroup.addItem("极坐标像素化", "polarPixellateLayer", polarPixellateLayer);
        layerGroup.addItem("圆形区域像素化", "pixellatePositionLayer", pixellatePositionLayer);
        layerGroup.addItem("网格化", "polkaDotLayer", polkaDotLayer);
        layerGroup.addItem("半色调效果", "pixellateLayer", halftoneLayer);
        layerGroup.addItem("黑白交叉阴影", "crosshatchLayer", crosshatchLayer);
        layerGroup.addItemLum("草图", "sketchLayer", luminanceLayer, sketchLayer, alphaShowLayer);
        layerGroup.addItemLum("草图阈值化", "thresholdSketchLayer", luminanceLayer, thresholdSketchLayer, alphaShowLayer);
        layerGroup.addItem("卡通", "toonLayer", toonLayer);
        layerGroup.addItem("平滑卡通", "smoothToonLayer", smoothToonLayer);
        layerGroup.addItem("压纹效果", "embossLayer", embossLayer);
        layerGroup.addItem("卡通阴影", "posterizeLayer", posterizeLayer);
        layerGroup.addItem("涡形失真", "swirlLayer", swirlLayer);
        layerGroup.addItem("凸起失真", "bulgeDistortionLayer", bulgeDistortionLayer);
        layerGroup.addItem("变形", "bulgeDistortionLayer", pinchDistortionLayer);
        layerGroup.addItem("拉伸变形", "stretchDistortionLayer", stretchDistortionLayer);
        layerGroup.addItem("球体折射", "sphereRefractionLayer", sphereRefractionLayer);
        layerGroup.addItem("球体反射", "sphereRefractionLayer", glassSphereLayer);
        layerGroup.addItem("渐晕效果", "vignetteLayer", vignetteLayer);
        layerGroup.addItem("油画效果", "kuwaharaLayer", kuwaharaLayer);
        layerGroup.addItem("模拟CGA颜色空间", "cgaColorspaceLayer", cgaColorspaceLayer);

        groups.add(layerGroup);
    }

    public static DataManager getInstance() {
        return DataManagerInstance.instance;
    }

    public LayerGroup getIndex(int index) {
        return groups.get(index);
    }

    public int getGroupCount() {
        return groups.size();
    }

    public void openCamera(boolean bFront) {
        aoceManager.openCamera(bFront);
    }

    public void closeCamer() {
        aoceManager.closeCamera();
    }

    public void clearGraph() {
        aoceManager.clearLayers();
    }

    public void copyTex(int textureId, int width, int height) {
        if (aoceManager != null) {
            aoceManager.showGL(textureId, width, height);
        }
    }

    public void initLayer(AssetManager assetManager, int groupIndex, int layerIndex) {
        LayerGroup layerGroup = groups.get(groupIndex);
        LayerItem layerItem = layerGroup.layers.get(layerIndex);
        if (layerItem.layerType == LayerType.normal) {
            aoceManager.initLayers(layerItem.getLayers(), false);
        } else if (layerItem.layerType == LayerType.edgeDetection) {
            ArrayList<IBaseLayer> layers = layerItem.getLayers();
            layers.add(0, luminanceLayer);
            layers.add(edgeBoxLayer.getLayer());
            layers.add(alphaShowLayer2);
            aoceManager.initLayers(layers, true);
        } else if (layerItem.layerType == LayerType.twoInput) {
            aoceManager.initLayers(layerItem.getLayers(), true);
        } else if (layerItem.layerType == LayerType.blend) {
            ArrayList<IBaseLayer> layers = layerItem.getLayers();
            aoceManager.initLayers(layers.get(0), inputBlendLayer);
            inputLayerLoadBitmap(assetManager, inputBlendLayer, blendImagePath);
        }
        if (layerItem.metadata == "lookupLayer") {
            String filtPath = "lookup_amatorka.png";
            if (layerItem.name.contains("MissEtikate")) {
                filtPath = "lookup_miss_etikate.png";
            }
            inputLayerLoadBitmap(assetManager, lookupLayer.getLookUpInputLayer(), filtPath);
        }
        if (layerItem.metadata == "softEleganceLayer") {
            inputLayerLoadBitmap(assetManager, softEleganceLayer.getLookUpInputLayer1(), "lookup_soft_elegance_1.png");
            inputLayerLoadBitmap(assetManager, softEleganceLayer.getLookUpInputLayer2(), "lookup_soft_elegance_2.png");
        }
    }

    public boolean inputLayerLoadBitmap(AssetManager assetManager, IInputLayer inputLayer, String filePath) {
        long ptr = IInputLayer.getCPtr(inputLayer);
        Bitmap bitmap = getBitmapFromAsset(assetManager, filePath);
        if (bitmap != null) {
            return JNIHelper.loadBitmap(ptr, bitmap);
        }
        return false;
    }

    public String getLayerName(int groupIndex, int layerIndex) {
        LayerGroup layerGroup = groups.get(groupIndex);
        LayerItem layerItem = layerGroup.layers.get(layerIndex);
        return layerItem.name;
    }

    public boolean haveParamet(int groupIndex, int layerIndex) {
        LayerGroup layerGroup = groups.get(groupIndex);
        LayerItem layerItem = layerGroup.layers.get(layerIndex);
        if (layerItem.metadata == null) {
            return false;
        }
        return AoceWrapper.getLayerMetadata(layerItem.metadata) != null;
    }

    public boolean havaMotion(int groupIndex, int layerIndex) {
        LayerGroup layerGroup = groups.get(groupIndex);
        LayerItem layerItem = layerGroup.layers.get(layerIndex);
        if (layerItem.metadata == "motionDetectorLayer") {
            return true;
        }
        return false;
    }

    public void setHandle(Handler handle) {
        this.handler = handle;
    }

    public Bitmap getBitmapFromAsset(AssetManager assetManager, String filePath) {
        InputStream istr = null;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }
        return bitmap;
    }

    public static Class<?> getPrimitive(final Class<?> primitiveType) {
        // does anyone know a better strategy than comparing names?
        if (Boolean.class.equals(primitiveType)) {
            return boolean.class;
        } else if (Float.class.equals(primitiveType)) {
            return float.class;
        } else if (Long.class.equals(primitiveType)) {
            return long.class;
        } else if (Integer.class.equals(primitiveType)) {
            return int.class;
        } else if (Short.class.equals(primitiveType)) {
            return short.class;
        } else if (Byte.class.equals(primitiveType)) {
            return byte.class;
        } else if (Double.class.equals(primitiveType)) {
            return double.class;
        } else if (Character.class.equals(primitiveType)) {
            return char.class;
        } else {
            return primitiveType;
        }
    }
}
