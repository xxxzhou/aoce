package aoce.android.library;

import android.graphics.Bitmap;

import aoce.android.library.xswig.*;


public class JNIHelper {
    static {
        System.loadLibrary("aoce_android");
    }
    public static native boolean loadBitmap(long inputLayer,Bitmap bitmap);
}

