package aoce.samples.aoceswigtest;

import androidx.appcompat.app.AppCompatActivity;
import aoce.android.library.wrapper.GLVideoRender;
import aoce.android.library.wrapper.IGLCopyTexture;
import aoce.android.library.xswig.*;
import aoce.samples.aoceswigtest.aocewrapper.AoceManager;

import android.Manifest;
import android.content.Intent;
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

    private int groupIndex = 0;
    private int layerIndex = 0;
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
        // LinearLayout linearLayout = viewGroup.findViewById(R.id.leLayout);

//        AoceWrapper.initPlatform();
//        AoceWrapper.loadAoce();
//        skinToneLayer = AoceWrapper.createSkinToneLayer();
//        skinToneParamet = skinToneLayer.getParamet();
//
//        ILMetadata layerMeta = AoceWrapper.getLayerMetadata("SkinToneLayer");
//        try {
//            bindMetaInterface(linearLayout, layerMeta, skinToneParamet, skinToneLayer, "updateParamet");
//        } catch (NoSuchMethodException e) {
//            e.printStackTrace();
//        } catch (InvocationTargetException e) {
//            e.printStackTrace();
//        } catch (IllegalAccessException e) {
//            e.printStackTrace();
//        }
        // 修改之后显示
        setContentView(viewGroup);

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView, this);

        Intent intent = getIntent();
        groupIndex = intent.getIntExtra("groupIndex",0);
        layerIndex = intent.getIntExtra("layerIndex",0);
    }

    @Override
    protected void onStart() {
        super.onStart();
        DataManager.getInstance().initLayer(getAssets(), groupIndex, layerIndex);
        DataManager.getInstance().openCamera(false);
    }
    @Override
    protected void onStop() {
        super.onStop();

        DataManager.getInstance().closeCamer();
        DataManager.getInstance().clearGraph();
    }

    @Override
    public void copyTex(int textureId, int width, int height) {
        DataManager.getInstance().copyTex(textureId, width, height);
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
//        LayerMetaTag layerMetaTag = (LayerMetaTag) seekBar.getTag();
//        Object object = null;
//        if(layerMetaTag.metadataType == LayerMetadataType.afloat){
//            ILFloatMetadata floatMetadata = (ILFloatMetadata)layerMetaTag.layerMeta;
//            object = (float)i/100.0f+floatMetadata.getMinValue();
//        } else if(layerMetaTag.metadataType == LayerMetadataType.aint){
//            ILIntMetadata floatMetadata = (ILIntMetadata)layerMetaTag.layerMeta;
//            object = i + floatMetadata.getMinValue();
//        }
//        if(object == null){
//            return;
//        }
//        try {
//            layerMetaTag.setMethod.invoke(layerMetaTag.layerObj,object);
//        } catch (IllegalAccessException e) {
//            e.printStackTrace();
//        } catch (InvocationTargetException e) {
//            e.printStackTrace();
//        }
//        updateLayerParamet();
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {
    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {

    }

    @Override
    public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
//        LayerMetaTag layerMetaTag = (LayerMetaTag) compoundButton.getTag();
//        // ILBoolMetadata boolMetadata = (ILBoolMetadata)layerMetaTag.layerMeta;
//        try {
//            layerMetaTag.setMethod.invoke(layerMetaTag.layerObj,b ? 1:0);
//        } catch (IllegalAccessException e) {
//            e.printStackTrace();
//        } catch (InvocationTargetException e) {
//            e.printStackTrace();
//        }
//        updateLayerParamet();
    }

    public void updateLayerParamet() {
        // skinToneLayer.updateParamet(skinToneParamet);
    }
}