package aoce.android.library;

import android.opengl.GLES30;
import android.opengl.GLSurfaceView;

import java.nio.ByteBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class GLVideoRender implements GLSurfaceView.Renderer {

    private GLSurfaceView surfaceView = null;
    private IGLCopyTexture copyTexture = null;
    private int width = 0;
    private int height = 0;
    int[] textureId = new int[1];
    int[] fboTexture = new int[1];
    // private bool bSurface = false;
    ByteBuffer pixelBuffer = null;
    public void init(GLSurfaceView glSurfaceView,IGLCopyTexture glCopyTexture) {
        surfaceView = glSurfaceView;
        copyTexture = glCopyTexture;
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    }

    @Override
    public void onSurfaceCreated(GL10 gl10, EGLConfig eglConfig) {
    }

    @Override
    public void onSurfaceChanged(GL10 gl10, int i, int i1) {
        width = i;
        height = i1;
        if (width > 1 && height > 1) {
            // 初始化一些数据,用于测试是否正确显示到窗口
            byte[] pixels = new byte[width * height * 4];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = 4 * (y * width + x);
                    // pixels[index] = (byte) 255;
                    pixels[index + 1] = (byte) 255;
                    pixels[index + 2] = (byte) 255;
                    pixels[index + 3] = (byte) 255;
                }
            }
            pixelBuffer = ByteBuffer.allocateDirect(width * height * 4);
            pixelBuffer.put(pixels).position(0);

            GLES30.glGenTextures(1, textureId, 0);
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId[0]);
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, width, height, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, pixelBuffer);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);

            GLES30.glGenFramebuffers(1, fboTexture, 0);
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboTexture[0]);
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, textureId[0], 0);
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        }
    }

    @Override
    public void onDrawFrame(GL10 gl10) {
        // 把vulkan计算后的纹理复制给gles纹理
        copyTexture.copyTex(textureId[0],width,height);

        GLES30.glViewport(0, 0, width, height);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        GLES30.glClearColor(1.0f, 0.0f, 0.f, 1.f);

        GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, fboTexture[0]);// Set framebuffer to read from
        GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, 0);// set window to draw to
        GLES30.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GLES30.GL_COLOR_BUFFER_BIT, GLES30.GL_NEAREST);
        // GLES30.glFlush();

        surfaceView.requestRender();
    }
}

