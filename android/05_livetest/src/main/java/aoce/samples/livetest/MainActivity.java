package aoce.samples.livetest;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import androidx.fragment.app.FragmentActivity;
import aoce.android.library.wrapper.GLVideoRender;
import aoce.android.library.wrapper.IGLCopyTexture;
import aoce.android.library.wrapper.IVKInitSurface;
import aoce.android.library.wrapper.VKVideoRender;
import aoce.samples.vulkantest.R;

public class MainActivity extends FragmentActivity implements IGLCopyTexture, IVKInitSurface, View.OnClickListener {
    static {
        System.loadLibrary("05_livetest");
        System.loadLibrary("agora-rtc-sdk-jni");
        // System.loadLibrary("vrapi");
    }

    private VKVideoRender vkVideoRender = null;
    private GLVideoRender glVideoRender = null;
    private Button btnOpen = null;
    private EditText uri = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnOpen = findViewById(R.id.btnJoin);
        btnOpen.setOnClickListener(this);
        uri = findViewById(R.id.roomName);

        initEngine(getApplicationContext());
        SurfaceView surfaceView = findViewById(R.id.vk_surface_view);
        vkVideoRender = new VKVideoRender();
        vkVideoRender.init(surfaceView, this);

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView, this);
    }

    @Override
    public void copyTex(int textureId, int width, int height) {
        glCopyTex(textureId, width, height);
    }

    @Override
    public void initSurface(Surface surface, int width, int height) {
        vkInitSurface(surface, width, height);
    }

    @Override
    public void onClick(View view) {
        joinRoom(uri.getText().toString());
    }

    public native void initEngine(Context context);

    public native void glCopyTex(int textureId, int width, int height);

    public native void vkInitSurface(Surface surface, int width, int height);

    public native void joinRoom(String uri);
}
