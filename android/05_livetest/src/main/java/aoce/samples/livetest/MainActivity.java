package aoce.samples.livetest;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
//import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import com.yanzhenjie.permission.AndPermission;
import com.yanzhenjie.permission.runtime.Permission;

import aoce.samples.vulkantest.R;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("05_livetest");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initEngine(getApplicationContext());
    }

    public native int initEngine(Context context);
}