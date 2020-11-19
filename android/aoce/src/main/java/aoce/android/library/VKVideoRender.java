package aoce.android.library;

import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class VKVideoRender implements SurfaceHolder.Callback {
    private IVKInitSurface initSurface = null;
    public void init(SurfaceView surface,IVKInitSurface vkInitSurface){
        initSurface = vkInitSurface;
        surface.getHolder().addCallback(this);
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        Log.i("aoce", "surfaceCreated: create");
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder,int format, int width, int height) {
        initSurface.initSurface(surfaceHolder.getSurface(),width,height);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
    }
}
