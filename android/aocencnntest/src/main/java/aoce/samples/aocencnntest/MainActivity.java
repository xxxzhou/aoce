package aoce.samples.aocencnntest;

import androidx.appcompat.app.AppCompatActivity;
import aoce.android.library.wrapper.GLVideoRender;
import butterknife.BindView;

import android.Manifest;
import android.content.pm.PackageManager;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.samples.aocencnntest.R;

import aoce.android.library.xswig.*;
import aoce.android.library.wrapper.*;
import aoce.samples.aocencnntest.AoceManager;
import butterknife.ButterKnife;

public class MainActivity extends AppCompatActivity implements IGLCopyTexture {
    private GLVideoRender glVideoRender = null;
    private AoceManager aoceManager = null;

    FloatingActionButton btnLoad = null;
    TextView textView = null;
    static {
        System.loadLibrary("ncnn");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView, this);

        btnLoad = findViewById(R.id.floatingActionButton);
        btnLoad.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // aoceManager.loadNet();
            }
        });

        if (checkSelfPermission(Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA},
                    1);
            return;
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        AoceWrapper.initPlatform();
        AoceWrapper.loadAoce();
        aoceManager = new AoceManager();
        aoceManager.initGraph();
        aoceManager.openCamera(true);
    }

    @Override
    protected void onStop() {
        super.onStop();
        aoceManager.closeCamera();
    }

    @Override
    public void copyTex(int textureId, int width, int height) {
       aoceManager.showGL(textureId, width, height);
    }
}