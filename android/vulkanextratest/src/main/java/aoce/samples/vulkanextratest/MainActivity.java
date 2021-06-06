package aoce.samples.vulkanextratest;

import androidx.fragment.app.FragmentActivity;
import aoce.android.library.wrapper.GLVideoRender;
import aoce.android.library.wrapper.IGLCopyTexture;

import android.Manifest;
import android.content.pm.PackageManager;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.widget.*;
import android.view.View;

// import com.android.colorpicker;

public class MainActivity extends FragmentActivity implements IGLCopyTexture, View.OnClickListener, SeekBar.OnSeekBarChangeListener {
    static {
        System.loadLibrary("vulkanextratest");
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

        initEngine();

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView,this);

        openLayer.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                enableLayer(isChecked);
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
        glCopyTex(textureId,width,height);
    }

    @Override
    public void onClick(View view) {
        int uid = cameraButton.isChecked()? 0 : 1;
        try {
            openCamera(uid);
        } catch (NumberFormatException e) {
        }
    }

    public void update(){
        boolean green = greenButton.isChecked();
        float luma = (float)lumaBar.getProgress()/lumaBar.getMax()*10.0f;
        float min =(float)minBar.getProgress()/minBar.getMax();
        float scale = (float)maxBar.getProgress()/maxBar.getMax()*100;
        float exponent = (float)scaleBar.getProgress()/scaleBar.getMax()*10;
        float dscale = (float)dscaleBar.getProgress()/dscaleBar.getMax()*100;
        updateParamet(green,luma,min,scale,exponent,dscale);
    }

    public native void initEngine();
    public native void glCopyTex(int textureId, int width, int height);
    public native void openCamera(int index);

    public native void enableLayer(boolean enable);
    public native void updateParamet(boolean green,float luma,float min,float scale,float exponent,float dscale);

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