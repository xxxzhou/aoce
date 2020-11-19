package aoce.samples.livetest;

import android.app.NativeActivity;
import aoce.samples.vulkantest.R;
import android.os.Bundle;
import io.agora.rtc2.RtcEngine2;

public class MainActivity2 extends NativeActivity {
    static {
        RtcEngine2.getSdkVersion();
        System.loadLibrary("05_livetest");
        System.loadLibrary("agora-rtc-sdk-jni");
    }
}