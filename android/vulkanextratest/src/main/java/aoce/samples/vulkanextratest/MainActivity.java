package aoce.samples.vulkanextratest;

import androidx.fragment.app.FragmentActivity;
import aoce.android.library.*;
import aoce.samples.vulkanextratest.R;

import android.Manifest;
import android.content.pm.PackageManager;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.Surface;
import android.view.SurfaceView;
import android.widget.*;
import android.view.View;

// import com.android.colorpicker;

public class MainActivity extends FragmentActivity implements IGLCopyTexture, View.OnClickListener {
    static {
        System.loadLibrary("vulkanextratest");
    }
    private GLVideoRender glVideoRender = null;
    private Button btnOpen = null;
    private EditText uri = null;
    private RadioButton radioButton = null;
    private Switch openLayer = null;

    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnOpen = findViewById(R.id.btnJoin);
        btnOpen.setOnClickListener(this);
        radioButton = findViewById(R.id.radioButton3);
        openLayer = findViewById(R.id.switch1);

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
    }

    @Override
    public void copyTex(int textureId, int width, int height) {
        glCopyTex(textureId,width,height);
    }

    @Override
    public void onClick(View view) {
        int uid = radioButton.isChecked()? 0 : 1;
        try {
            openCamera(uid);
        } catch (NumberFormatException e) {
        }
    }

    public native void initEngine();
    public native void glCopyTex(int textureId, int width, int height);
    public native void openCamera(int index);
}