package aoce.android.library;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import aoce.android.library.xswig.*;

public class JNIHelper {
    static {
        System.loadLibrary("aoce_android");
    }

    public static native boolean loadBitmap(long inputLayer, Bitmap bitmap);

    public static native void setVideoFrameData(long videoFrame, long data);

    public static native void setAgoraContext(long agoraContext, Context context);

    public static boolean loadBitmap(IInputLayer inputLayer, Bitmap bitmap) {
        return loadBitmap(IInputLayer.getCPtr(inputLayer), bitmap);
    }

    public static void setVideoFrameData(VideoFrame videoFrame, SWIGTYPE_p_unsigned_char data) {
        setVideoFrameData(VideoFrame.getCPtr(videoFrame), SWIGTYPE_p_unsigned_char.getCPtr(data));
    }

    public static void setAgoraContext(AgoraContext agoraContext, Context context) {
        setAgoraContext(AgoraContext.getCPtr(agoraContext), context);
    }

}

