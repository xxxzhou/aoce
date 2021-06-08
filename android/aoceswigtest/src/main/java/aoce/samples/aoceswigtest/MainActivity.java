package aoce.samples.aoceswigtest;

import androidx.fragment.app.FragmentActivity;
import aoce.android.library.xswig.AoceWrapper;
import aoce.android.library.wrapper.GLVideoRender;
import aoce.android.library.wrapper.IGLCopyTexture;
import aoce.android.library.xswig.IBaseLayer;
import aoce.android.library.xswig.IChromaKeyLayer;
import aoce.samples.aoceswigtest.aocewrapper.AoceManager;
import aoce.android.library.xswig.*;

import android.Manifest;
import android.content.pm.PackageManager;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.ViewGroup;
import android.view.ViewManager;
import android.widget.*;
import android.view.View;

import java.util.List;
import java.util.ArrayList;

public class MainActivity extends FragmentActivity implements IGLCopyTexture, View.OnClickListener, SeekBar.OnSeekBarChangeListener {
    static {
        System.loadLibrary("aoce_swig_java");
    }
    private GLVideoRender glVideoRender = null;
    private Button btnOpen = null;
    private EditText uri = null;
    private RadioButton cameraButton = null;
    private CheckBox openLayer = null;
    private RadioButton greenButton = null;
    private RadioButton blueButton = null;
    private SeekBar lumaBar = null;
    private SeekBar minBar = null;
    private SeekBar maxBar = null;
    private SeekBar scaleBar = null;
    private SeekBar dscaleBar = null;

    private AoceManager aoceManager = null;
    private IChromaKeyLayer chromaKeyLayer = null;

    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnOpen = findViewById(R.id.btnJoin);
        btnOpen.setOnClickListener(this);
        cameraButton = findViewById(R.id.radioButton3);
        openLayer = findViewById(R.id.checkBox);
        greenButton = findViewById(R.id.radioButton6);
        blueButton = findViewById(R.id.radioButton5);
        lumaBar = findViewById(R.id.seekBar);
        minBar = findViewById(R.id.seekBar2);
        maxBar = findViewById(R.id.seekBar3);
        scaleBar = findViewById(R.id.seekBar4);
        dscaleBar = findViewById(R.id.seekBar5);

        if (checkSelfPermission( Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA},
                    PERMISSION_REQUEST_CODE_CAMERA);
            return;
        }
        AoceWrapper.initPlatform();
        AoceWrapper.loadAoce();
        aoceManager = new AoceManager();
        aoceManager.initGraph();

        chromaKeyLayer = AoceWrapper.createChromaKeyLayer();
        ArrayList<IBaseLayer> layers = new ArrayList<IBaseLayer>();
        layers.add(chromaKeyLayer.getLayer());
        aoceManager.initLayers(layers,false);

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView,this);

        openLayer.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                //enableLayer(isChecked);
            }
        });
        greenButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                update();
            }
        });
        blueButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                update();
            }
        });
        lumaBar.setOnSeekBarChangeListener(this);
        minBar.setOnSeekBarChangeListener(this);
        maxBar.setOnSeekBarChangeListener(this);
        scaleBar.setOnSeekBarChangeListener(this);
    }

    @Override
    public void copyTex(int textureId, int width, int height) {
        //glCopyTex(textureId,width,height);
        if(aoceManager != null){
            aoceManager.showGL(textureId,width,height);
        }
    }

    @Override
    public void onClick(View view) {
        try {
            aoceManager.openCamera(!cameraButton.isChecked());
        } catch (NumberFormatException e) {
        }
    }

    public void update(){
        ChromaKeyParamet cp = chromaKeyLayer.getParamet();
        vec3 green = new vec3();
        green.setX(0.15f);
        green.setY(0.6f);
        vec3 blue = new vec3();
        blue.setX(0.15f);
        blue.setY(0.2f);
        blue.setZ(0.8f);
        vec3 chromeColor = greenButton.isChecked() ? green:blue;
        cp.setChromaColor(chromeColor);
        cp.setLumaMask((float)lumaBar.getProgress()/lumaBar.getMax()*10.0f);
        cp.setAlphaCutoffMin((float)minBar.getProgress()/minBar.getMax());
        cp.setAlphaScale((float)maxBar.getProgress()/maxBar.getMax()*100);
        cp.setAlphaExponent((float)scaleBar.getProgress()/scaleBar.getMax()*10);
        cp.setDespillScale((float)dscaleBar.getProgress()/dscaleBar.getMax()*100);
        chromaKeyLayer.updateParamet(cp);
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
        update();
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {
    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {
        update();
    }
}