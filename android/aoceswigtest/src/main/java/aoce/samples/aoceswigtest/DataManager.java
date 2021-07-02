package aoce.samples.aoceswigtest;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.function.BiFunction;

import aoce.android.library.JNIHelper;
import aoce.android.library.xswig.*;
import aoce.samples.aoceswigtest.aocewrapper.AoceManager;

public class DataManager {
    private AoceManager aoceManager = null;

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


    private IBaseLayer alphaShowLayer = null;

    public class LayerItem {
        public String name = "";
        public String metadata = "";
        public ArrayList<IBaseLayer> layers = new ArrayList<>();
    }

    public class LayerGroup {
        public String title = "";
        public ArrayList<LayerItem> layers = new ArrayList<LayerItem>();

        public void addItem(String itemName) {
            LayerItem item = new LayerItem();
            item.name = itemName;
            layers.add(item);
        }

        public void addItem(String itemName, String metadataname, ILayer layer) {
            addItem(itemName, metadataname, layer.getLayer());
        }

        public void addItem(String itemName, IBaseLayer... baseLayers) {
            addItem(itemName, null, baseLayers);
        }

        public void addItem(String itemName, String metadataname, IBaseLayer... baseLayers) {
            LayerItem item = new LayerItem();
            item.name = itemName;
            item.metadata = metadataname;
            for (int i = 0; i < baseLayers.length; i++) {
                item.layers.add(baseLayers[i]);
            }
            layers.add(item);
        }
    }

    private ArrayList<LayerGroup> groups = new ArrayList<LayerGroup>();

    private static class DataManagerInstance {
        private static final DataManager instance = new DataManager();
    }

    private DataManager() {
        System.loadLibrary("aoce_swig_java");

        AoceWrapper.initPlatform();
        AoceWrapper.loadAoce();
        aoceManager = new AoceManager();
        aoceManager.initGraph();
        initColorAdjustmentsLayer();
        initImageProcessingLayer();
    }

    private void initColorAdjustmentsLayer() {
        alphaShowLayer = AoceWrapper.createAlphaShowLayer();

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
        layerGroup.addItem("Amatorka映射", "lookupLayer", lookupLayer);
        layerGroup.addItem("MissEtikate映射", "lookupLayer", lookupLayer);
        layerGroup.addItem("SoftElegance", "softEleganceLayer", softEleganceLayer);
        layerGroup.addItem("肤色调整滤镜", "skinToneLayer", skinToneLayer);
        layerGroup.addItem("反转图像", "colorInverLayer", colorInverLayer);
        layerGroup.addItem("灰度", "luminanceLayer", luminanceLayer, alphaShowLayer);
        layerGroup.addItem("基于亮度的单色", "monochromeLayer", monochromeLayer);
        layerGroup.addItem("基于亮度的混合", "falseColorLayer", falseColorLayer);
        layerGroup.addItem("调整雾度", "hazeLayer", hazeLayer);
        layerGroup.addItem("棕褐色调", "sepiaLayer", sepiaLayer);
        layerGroup.addItem("基于亮度的阈值", "luminanceThresholdLayer", luminanceThresholdLayer);
        layerGroup.addItem("基于周边亮度的阈值", "adaptiveThresholdLayer", adaptiveThresholdLayer);
        layerGroup.addItem("基于周边平均亮度的自适应阈值", "averageLuminanceThresholdLayer", averageLuminanceThresholdLayer);
        layerGroup.addItem("直方图", "historamlayer", historamlayer);
        layerGroup.addItem("色度扣像", "chromaKeyLayer", chromaKeyLayer);
        layerGroup.addItem("色相/饱和度/亮度", "hsbLayer", hsbLayer);
        groups.add(layerGroup);
    }

    private void initImageProcessingLayer() {
        LayerGroup layerGroup = new LayerGroup();
        layerGroup.title = "图像处理";
        layerGroup.addItem("锐化图像");
        layerGroup.addItem("模糊蒙版");
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
        aoceManager.initLayers(layerItem.layers, false);
        if (layerItem.metadata == "lookupLayer") {
            String filtPath = "lookup_amatorka.png";
            if(layerItem.name.contains("MissEtikate")){
                filtPath = "lookup_miss_etikate.png";
            }
            inputLayerLoadBitmap(assetManager,lookupLayer.getLookUpInputLayer(),filtPath);
        }
        if(layerItem.metadata == "softEleganceLayer"){
            inputLayerLoadBitmap(assetManager,softEleganceLayer.getLookUpInputLayer1(),"lookup_soft_elegance_1.png");
            inputLayerLoadBitmap(assetManager,softEleganceLayer.getLookUpInputLayer2(),"lookup_soft_elegance_2.png");
        }
    }

    public boolean inputLayerLoadBitmap(AssetManager assetManager,IInputLayer inputLayer,String filePath){
        long ptr = IInputLayer.getCPtr(inputLayer);
        Bitmap bitmap = getBitmapFromAsset(assetManager,filePath);
        if(bitmap != null) {
            return JNIHelper.loadBitmap(ptr, bitmap);
        }
        return false;
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

    public void BindMetaInterface(Context context, LinearLayout linearLayout, ILMetadata clMeta, Object obj, Object parentObj, String parametName)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        LinearLayout.LayoutParams lylp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        LinearLayout linearLayout1 = new LinearLayout(context);
        linearLayout1.setLayoutParams(lylp);
        if (clMeta.getLayerType() == LayerMetadataType.agroup) {
            // 垂直添加
            linearLayout1.setOrientation(LinearLayout.VERTICAL);
            ILGroupMetadata groupMetadata = AoceWrapper.getLGroupMetadata(clMeta);
            int lcount = groupMetadata.getCount();
            for (int i = 0; i < lcount; i++) {
                ILMetadata ccMeta = groupMetadata.getLMetadata(i);
                String cparametName = ccMeta.getParametName();
                String jparametName = cparametName.substring(0, 1).toUpperCase() + cparametName.substring(1);
                String methodStr = "get" + jparametName;
                Method getMethod = obj.getClass().getMethod(methodStr);
                Object childObj = getMethod.invoke(obj);
                BindMetaInterface(context, linearLayout1, ccMeta, childObj, obj, "set" + jparametName);
            }
            linearLayout.addView(linearLayout1);
        } else {
            // 水平添加
            linearLayout1.setOrientation(LinearLayout.HORIZONTAL);
            LayerMetaTag metaTag = new LayerMetaTag();
            metaTag.metadataType = clMeta.getLayerType();
            metaTag.layerObj = parentObj;
            // 把obj的类型拆箱
            metaTag.setMethod = parentObj.getClass().getMethod(parametName, getPrimitive(obj.getClass()));
            if (clMeta.getLayerType() == LayerMetadataType.abool) {
                ILBoolMetadata boolMetadata = AoceWrapper.getLBoolMetadata(clMeta);
                CheckBox box = new CheckBox(context);
                box.setText(boolMetadata.getText());
                box.setLayoutParams(lylp);
                boolean value = false;
                if (obj.getClass() == Boolean.class || obj.getClass() == boolean.class) {
                    value = (boolean) obj;
                } else {
                    value = (int) obj != 0;
                }
                box.setChecked(value);
                metaTag.layerMeta = boolMetadata;
                box.setTag(metaTag);
                // box.setOnCheckedChangeListener(this);
                linearLayout1.addView(box);
                linearLayout.addView(linearLayout1);
            } else if (clMeta.getLayerType() == LayerMetadataType.aint) {
                ILIntMetadata intMetadata = AoceWrapper.getLIntMetadata(clMeta);
                TextView textView = new TextView(context);
                textView.setText(clMeta.getText());
                linearLayout1.addView(textView);
                SeekBar seekBar = new SeekBar(context);
                seekBar.setMax(intMetadata.getMaxValue() - intMetadata.getMinValue());
                seekBar.setProgress((int) obj - intMetadata.getMinValue());
                seekBar.setLayoutParams(lylp);
                seekBar.setTag(metaTag);
                // seekBar.setOnSeekBarChangeListener(context);
                linearLayout1.addView(seekBar);
                linearLayout.addView(linearLayout1);
            } else if (clMeta.getLayerType() == LayerMetadataType.afloat) {
                ILFloatMetadata floatMetadata = AoceWrapper.getLFloatMetadata(clMeta);
                TextView textView = new TextView(context);
                textView.setText(clMeta.getText());
                linearLayout1.addView(textView);
                SeekBar seekBar = new SeekBar(context);
                float range = floatMetadata.getMaxValue() - floatMetadata.getMinValue();
                float cv = (float) obj - floatMetadata.getMinValue();
                seekBar.setMax((int) (range * 100));
                seekBar.setProgress((int) (cv * 100));
                seekBar.setLayoutParams(lylp);
                metaTag.layerMeta = floatMetadata;
                seekBar.setTag(metaTag);
                // seekBar.setOnSeekBarChangeListener(this);
                linearLayout1.addView(seekBar);
                linearLayout.addView(linearLayout1);
            }
        }
    }

}
