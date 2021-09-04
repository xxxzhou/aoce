package aoce.samples.aocencnntest;

import java.util.ArrayList;
import java.util.List;

import aoce.android.library.xswig.*;

public class AoceManager extends IVideoDeviceObserver {
    private IPipeGraph pipeGraph = null;
    private LayerFactory layerFactory = null;
    private IInputLayer inputLayer = null;
    private IOutputLayer outputLayer = null;
    private IYUVLayer yuv2RGBALayer = null;   ;
    private ITransposeLayer transposeLayerNcnn = null;
    private ITransposeLayer transposeLayer = null;
    private IFlipLayer flipLayer = null;
    private IReSizeLayer reSizeLayer = null;
    private IBaseLayer extraLayer = null;
    private IVideoDevice videoDevice = null;
    private VideoFormat videoFormat = null;
    private GLOutGpuTex gpuTex = new GLOutGpuTex();

    private IFaceDetector faceDetector = null;
    private IFaceKeypointDetector faceKeypointDetector = null;
    private IBaseLayer ncnnInLayer = null;
    private INcnnInCropLayer ncnnInCropLayer = null;
    private IDrawRectLayer drawRectLayer = null;
    private IDrawPointsLayer drawPointsLayer = null;

    private int width = 1280;
    private int height = 720;

    public void initGraph() {
        pipeGraph = AoceWrapper.getPipeGraphFactory(GpuType.vulkan).createGraph();
        layerFactory = AoceWrapper.getLayerFactory(GpuType.vulkan);
        inputLayer = layerFactory.createInput();
        outputLayer = layerFactory.createOutput();
        yuv2RGBALayer = layerFactory.createYUV2RGBA();
        transposeLayerNcnn = layerFactory.createTranspose();
        transposeLayer = layerFactory.createTranspose();
        flipLayer = layerFactory.createFlip();
        reSizeLayer = layerFactory.createSize();

        faceDetector = AoceWrapper.createFaceDetector();
        faceKeypointDetector = AoceWrapper.createFaceKeypointDetector();
        ncnnInLayer = AoceWrapper.createNcnnInLayer();
        ncnnInCropLayer = AoceWrapper.createNcnnInCropLayer();
        drawRectLayer = AoceWrapper.createDrawRectLayer();
        drawPointsLayer = AoceWrapper.createDrawPointsLayer();

        faceDetector.setDraw(5,new vec4(0.0f,1.0f,0.0f,1.0f));
        faceKeypointDetector.setDraw(5,new vec4(1.0f,0.0f,0.0f,1.0f));

        FlipParamet fp = flipLayer.getParamet();
        fp.setBFlipX(0);
        fp.setBFlipY(1);
        flipLayer.updateParamet(fp);

        TransposeParamet tpNcnn = transposeLayerNcnn.getParamet();
        tpNcnn.setBFlipX(1);
        tpNcnn.setBFlipY(0);
        transposeLayerNcnn.updateParamet(tpNcnn);

        TransposeParamet tp = transposeLayer.getParamet();
        tp.setBFlipX(1);
        tp.setBFlipY(0);
        transposeLayer.updateParamet(tpNcnn);

        OutputParamet op = outputLayer.getParamet();
        op.setBGpu(1);
        op.setBCpu(0);
        outputLayer.updateParamet(op);

        ReSizeParamet reSizeParamet = reSizeLayer.getParamet();
        reSizeParamet.setNewHeight(480);
        reSizeParamet.setNewWidth(480);
        reSizeLayer.updateParamet(reSizeParamet);

        initLayers();
        loadNet();
    }

    public void openCamera() {
        openCamera(false);
    }

    public void openCamera(boolean bFront) {
        if (videoDevice != null && videoDevice.bOpen()) {
            videoDevice.close();
        }
        int deviceCount = AoceWrapper.getVideoManager(CameraType.and_camera2).getDeviceCount();
        for (int i = 0; i < deviceCount; i++) {
            videoDevice = AoceWrapper.getVideoManager(CameraType.and_camera2).getDevice(i);
            if (videoDevice.back() != bFront) {
                break;
            }
        }
        int formatIndex = videoDevice.findFormatIndex(width, height);
        if (formatIndex < 0) {
            formatIndex = 0;
        }
        videoDevice.setFormat(formatIndex);
        videoDevice.open();

        videoFormat = videoDevice.getSelectFormat();
        width = videoFormat.getWidth();
        height = videoFormat.getHeight();

        videoDevice.setObserver(this);
    }

    public void closeCamera() {
        if (videoDevice != null) {
            videoDevice.close();
        }
    }

    public void loadNet(){
        faceDetector.initNet(ncnnInLayer,drawRectLayer);
        faceDetector.setFaceKeypointObserver(ncnnInCropLayer);
        faceKeypointDetector.initNet(ncnnInCropLayer,drawPointsLayer);
    }

    public void initLayers() {
        pipeGraph.clear();
        extraLayer = pipeGraph.addNode(inputLayer).addNode(yuv2RGBALayer);//.addNode(transposeLayerNcnn);
        pipeGraph.addNode(ncnnInLayer);
        pipeGraph.addNode(ncnnInCropLayer);
        yuv2RGBALayer.getLayer().addLine(ncnnInLayer);
        yuv2RGBALayer.getLayer().addLine(ncnnInCropLayer.getLayer());
        yuv2RGBALayer.getLayer().addNode(drawRectLayer).addNode(drawPointsLayer).addNode(transposeLayerNcnn)
                .addNode(outputLayer);//.addNode(flipLayer)
    }

    public void clearLayers() {
        pipeGraph.clear();
    }

    @Override
    public void onVideoFrame(VideoFrame frame) {
        if (AoceWrapper.getYuvIndex(frame.getVideoType()) < 0) {
            yuv2RGBALayer.getLayer().setVisable(false);
        } else if (videoFormat != null) {
            if (yuv2RGBALayer.getParamet().getType() != frame.getVideoType()) {
                yuv2RGBALayer.getLayer().setVisable(true);
                YUVParamet yp = yuv2RGBALayer.getParamet();
                yp.setType(frame.getVideoType());
                yuv2RGBALayer.updateParamet(yp);
            }
        }
        inputLayer.inputCpuData(frame);
        pipeGraph.run();
    }

    public void showGL(int textureId, int width, int height) {
        gpuTex.setImage(textureId);
        gpuTex.setWidth(width);
        gpuTex.setHeight(height);
        outputLayer.outGLGpuTex(gpuTex);
    }
}
