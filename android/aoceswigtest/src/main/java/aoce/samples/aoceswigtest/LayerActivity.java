package aoce.samples.aoceswigtest;

import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentManager;
import aoce.android.library.wrapper.GLVideoRender;
import aoce.android.library.wrapper.IGLCopyTexture;
import aoce.android.library.xswig.*;
import aoce.samples.aoceswigtest.aocewrapper.AoceManager;
import aoce.samples.aoceswigtest.ui.layerparamet.ParametFragment;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.opengl.GLSurfaceView;
import android.os.Bundle;

import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.CompoundButton;
import android.widget.PopupWindow;
import android.widget.SeekBar;
import android.widget.TextView;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

public class LayerActivity extends AppCompatActivity implements IGLCopyTexture {

    private GLVideoRender glVideoRender = null;
    @BindView(R.id.floatingActionButton)
    FloatingActionButton btnParametWin = null;
    @BindView(R.id.floatingActionButton1)
    FloatingActionButton btnBack = null;
    @BindView(R.id.textView11)
    TextView textView = null;
    @BindView(R.id.textView12)
    TextView textMotion = null;
    @BindView(R.id.textView14)
    TextView textLayer = null;

    private int groupIndex = 0;
    private int layerIndex = 0;

    private boolean bParamet = false;
    private boolean bMotion = false;

    private Handler handler = new Handler(){
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what){
                case DataManager.MONION_UPDATE:
                    String text = "X:"+msg.arg1+" Y:"+msg.arg2;
                    textMotion.setText(text);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ViewGroup viewGroup = (ViewGroup) LayoutInflater.from(this).inflate(R.layout.activity_main2, null);
        setContentView(viewGroup);
        ButterKnife.bind(this);

        glVideoRender = new GLVideoRender();
        GLSurfaceView glSurfaceView = findViewById(R.id.es_surface_view);
        glVideoRender.init(glSurfaceView, this);

        Intent intent = getIntent();
        groupIndex = intent.getIntExtra("groupIndex", 0);
        layerIndex = intent.getIntExtra("layerIndex", 0);

        String layerName = DataManager.getInstance().getLayerName(groupIndex, layerIndex);
        textLayer.setText(layerName);

        bParamet = DataManager.getInstance().haveParamet(groupIndex, layerIndex);
        if (!bParamet) {
            textView.setVisibility(View.GONE);
            btnParametWin.setVisibility(View.GONE);
        }
        bMotion =  DataManager.getInstance().havaMotion(groupIndex, layerIndex);
        if(!bMotion){
            textMotion.setVisibility(View.GONE);
        }
        DataManager.getInstance().setHandle(handler);
    }

    @OnClick({R.id.floatingActionButton})
    public void onClick(View view) {
        FragmentManager fm = getSupportFragmentManager();
        ParametFragment lp = ParametFragment.newInstance(groupIndex, layerIndex);
        lp.show(fm, "parmamet win");
    }

    @OnClick({R.id.floatingActionButton1})
    public void onBack(View view) {
        finish();
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
}