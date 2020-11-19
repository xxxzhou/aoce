package aoce.samples.livetest;

        import android.view.Surface;
        import android.view.SurfaceHolder;
        import android.view.SurfaceView;

public class VKVideoRender implements SurfaceHolder.Callback {
    public void init(SurfaceView surface){
        surface.getHolder().addCallback(this);
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {

    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder,int format, int width, int height) {
        initSurface(surfaceHolder.getSurface(),width,height);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {

    }

    public native int initSurface(Surface surface,int width,int height);
}
