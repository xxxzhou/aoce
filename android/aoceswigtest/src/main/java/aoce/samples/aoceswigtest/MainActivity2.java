package aoce.samples.aoceswigtest;

import androidx.appcompat.app.AppCompatActivity;
import aoce.android.library.wrapper.GLVideoRender;
import aoce.android.library.wrapper.IGLCopyTexture;
import aoce.android.library.xswig.*;
import aoce.samples.aoceswigtest.aocewrapper.AoceManager;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import java.lang.reflect.*;
import java.util.ArrayList;

public class MainActivity2 extends AppCompatActivity implements IGLCopyTexture, SeekBar.OnSeekBarChangeListener, CompoundButton.OnCheckedChangeListener {
    static {
        System.loadLibrary("aoce_swig_java");
    }

    private GLVideoRender glVideoRender = null;
    private AoceManager aoceManager = null;
    private IChromaKeyLayer chromaKeyLayer = null;
    private ISkinToneLayer skinToneLayer = null;
    private SkinToneParamet skinToneParamet = null;

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

    public void bindMetaInterface(LinearLayout linearLayout, ILMetadata clMeta, Object obj, Object parentObj, String parametName)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        LinearLayout.LayoutParams lylp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        LinearLayout linearLayout1 = new LinearLayout(this);
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
                bindMetaInterface(linearLayout1, ccMeta, childObj, obj, "set" + jparametName);
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
                CheckBox box = new CheckBox(this);
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
                box.setOnCheckedChangeListener(this);
                linearLayout1.addView(box);
                linearLayout.addView(linearLayout1);
            } else if (clMeta.getLayerType() == LayerMetadataType.aint) {
                ILIntMetadata intMetadata = AoceWrapper.getLIntMetadata(clMeta);
                TextView textView = new TextView(this);
                textView.setText(clMeta.getText());
                linearLayout1.addView(textView);
                SeekBar seekBar = new SeekBar(this);
                seekBar.setMax(intMetadata.getMaxValue() - intMetadata.getMinValue());
                seekBar.setProgress((int) obj - intMetadata.getMinValue());
                seekBar.setLayoutParams(lylp);
                seekBar.setTag(metaTag);
                seekBar.setOnSeekBarChangeListener(this);
                linearLayout1.addView(seekBar);
                linearLayout.addView(linearLayout1);
            } else if (clMeta.getLayerType() == LayerMetadataType.afloat) {
                ILFloatMetadata floatMetadata = AoceWrapper.getLFloatMetadata(clMeta);
                TextView textView = new TextView(this);
                textView.setText(clMeta.getText());
                linearLayout1.addView(textView);
                SeekBar seekBar = new SeekBar(this);
                float range = floatMetadata.getMaxValue() - floatMetadata.getMinValue();
                float cv = (float) obj - floatMetadata.getMinValue();
                seekBar.setMax((int) (range * 100));
                seekBar.setProgress((int) (cv * 100));
                seekBar.setLayoutParams(lylp);
                metaTag.layerMeta = floatMetadata;
                seekBar.setTag(metaTag);
                seekBar.setOnSeekBarChangeListener(this);
                linearLayout1.addView(seekBar);
                linearLayout.addView(linearLayout1);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (checkSelfPermission(Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA},
                    1);
            return;
        }

        ViewGroup viewGroup = (ViewGroup) LayoutInflater.from(this).inflate(R.layout.activity_main2, null);
        LinearLayout linearLayout = viewGroup.findViewById(R.id.leLayout);

        AoceWrapper.initPlatform();
        AoceWrapper.loadAoce();
        skinToneLayer = AoceWrapper.createSkinToneLayer();
        skinToneParamet = skinToneLayer.getParamet();

        ILMetadata layerMeta = AoceWrapper.getLayerMetadata("SkinToneLayer");
        try {
            bindMetaInterface(linearLayout, layerMeta, skinToneParamet, skinToneLayer, "updateParamet");
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        // 修改之后显示
        setContentView(viewGroup);

        aoceManager = new AoceManager();
        aoceManager.initGraph();

        ArrayList<IBaseLayer> layers = new ArrayList<IBaseLayer>();
        layers.add(skinToneLayer.getLayer());
        aoceManager.initLayers(layers, false);

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView, this);

        aoceManager.openCamera(false);
    }

    @Override
    public void copyTex(int textureId, int width, int height) {
        //glCopyTex(textureId,width,height);
        if (aoceManager != null) {
            aoceManager.showGL(textureId, width, height);
        }
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
        LayerMetaTag layerMetaTag = (LayerMetaTag) seekBar.getTag();
        Object object = null;
        if(layerMetaTag.metadataType == LayerMetadataType.afloat){
            ILFloatMetadata floatMetadata = (ILFloatMetadata)layerMetaTag.layerMeta;
            object = (float)i/100.0f+floatMetadata.getMinValue();
        } else if(layerMetaTag.metadataType == LayerMetadataType.aint){
            ILIntMetadata floatMetadata = (ILIntMetadata)layerMetaTag.layerMeta;
            object = i + floatMetadata.getMinValue();
        }
        if(object == null){
            return;
        }
        try {
            layerMetaTag.setMethod.invoke(layerMetaTag.layerObj,object);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }
        updateLayerParamet();
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {
    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {

    }

    @Override
    public void onCheckedChanged(CompoundButton compoundButton, boolean b){
        LayerMetaTag layerMetaTag = (LayerMetaTag) compoundButton.getTag();
        // ILBoolMetadata boolMetadata = (ILBoolMetadata)layerMetaTag.layerMeta;
        try {
            layerMetaTag.setMethod.invoke(layerMetaTag.layerObj,b ? 1:0);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }
        updateLayerParamet();
    }

    public void updateLayerParamet(){
        skinToneLayer.updateParamet(skinToneParamet);
    }
}