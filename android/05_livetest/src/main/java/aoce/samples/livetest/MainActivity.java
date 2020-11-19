package aoce.samples.livetest;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.os.Bundle;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.EditText;

import androidx.fragment.app.FragmentActivity;
import aoce.samples.vulkantest.R;

public class MainActivity extends FragmentActivity {
    static {
        System.loadLibrary("05_livetest");
        System.loadLibrary("agora-rtc-sdk-jni");
    }

    private VKVideoRender videoRender = null;
    private Button btnJoin = null;
    private EditText roomTex = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initEngine(getApplicationContext());
        SurfaceView surfaceView = findViewById(R.id.surface_view);
        btnJoin = findViewById(R.id.btnJoin);
        roomTex = findViewById(R.id.roomName);
        videoRender = new VKVideoRender();
        videoRender.init(surfaceView);
    }

    public native int initEngine(Context context);
}