package aoce.samples.livetest;

import androidx.fragment.app.FragmentActivity;
import aoce.samples.vulkantest.R;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.SurfaceView;

public class MainActivity3 extends FragmentActivity {
    static {
        System.loadLibrary("05_livetest");
        System.loadLibrary("agora-rtc-sdk-jni");
    }

    private GLVideoRender videoRender = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main3);
        initRoom(getApplicationContext());

        videoRender = new GLVideoRender();
        GLSurfaceView surfaceView = findViewById(R.id.surface_view);
        videoRender.init(surfaceView);
    }

    public native int initRoom(Context context);
}